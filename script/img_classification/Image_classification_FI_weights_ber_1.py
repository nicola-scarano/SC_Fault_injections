import argparse
import datetime
import json
import os
import time

import torch
from torch import distributed as dist
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torchdistill.common import file_util, yaml_util, module_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, init_distributed_mode, load_ckpt, save_ckpt, set_seed
from torchdistill.core.distillation import get_distillation_box
from torchmetrics.classification import MulticlassF1Score, MulticlassRecall, MulticlassPrecision
from torchdistill.core.training import get_training_box
from torchdistill.datasets import util
from torchdistill.eval.classification import compute_accuracy
from torchdistill.misc.log import setup_log_file, SmoothedValue, MetricLogger

from sc2bench.analysis import check_if_analyzable
from sc2bench.common.config_util import overwrite_config
from sc2bench.models.backbone import check_if_updatable
from sc2bench.models.registry import load_classification_model
from sc2bench.models.wrapper import get_wrapped_classification_model

import pytorchfi
from pytorchfi import core
from pytorchfi import neuron_error_models
from pytorchfi import weight_error_models

from pytorchfi.core import FaultInjection
import torchvision.transforms as transforms
import torchvision

from pytorchfi.FI_Weights_classification import FI_report_classifier
from pytorchfi.FI_Weights_classification import FI_framework
from pytorchfi.FI_Weights_classification import FI_manager 
from pytorchfi.FI_Weights_classification import DatasetSampling 

from torch.utils.data import DataLoader, Subset

logger = def_logger.getChild(__name__)
# comment this line, otherwise the fault injections will collapse due to leaking memory produced by 'file_system
#torch.multiprocessing.set_sharing_strategy('file_system')
import logging

def get_argparser():
    parser = argparse.ArgumentParser(description='Supervised compression for image classification tasks')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--json', help='json string to overwrite config')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--log', help='log file path')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--seed', type=int, help='seed in random number generator')
    parser.add_argument('-test_only', action='store_true', help='only test the models')
    parser.add_argument('-student_only', action='store_true', help='test the student model only')
    parser.add_argument('-no_dp_eval', action='store_true',
                        help='perform evaluation without DistributedDataParallel/DataParallel')
    parser.add_argument('-log_config', action='store_true', help='log config')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('-adjust_lr', action='store_true',
                        help='multiply learning rate by number of distributed processes (world_size)')
    parser.add_argument('--fsim_config', help='Yaml file path fsim config')
    parser.add_argument('--ber', help='Error Rate')
    parser.add_argument('--bitloc', help='Bit location')
    parser.add_argument('--layr_idx', help='Bit location')
    return parser


def load_model(model_config, device, distributed):
    if 'classification_model' in model_config:
        return load_classification_model(model_config, device, distributed)
    return get_wrapped_classification_model(model_config, device, distributed)



@torch.inference_mode()
def evaluate(model_wo_ddp, data_loader, device, device_ids, distributed, no_dp_eval=False,
             log_freq=100, title=None, header='Test:', fsim_enabled=False, Fsim_setup:FI_manager = None):
    model = model_wo_ddp.to(device)
    if distributed and not no_dp_eval:
        model = DistributedDataParallel(model_wo_ddp, device_ids=device_ids)
    elif device.type.startswith('cuda') and not no_dp_eval:
        model = DataParallel(model_wo_ddp, device_ids=device_ids)
    elif hasattr(model, 'use_cpu4compression'):
        model.use_cpu4compression()

    if title is not None:
        logger.info(title)

    model.eval()

    analyzable = check_if_analyzable(model_wo_ddp)
    metric_logger = MetricLogger(delimiter='  ')

    im=0
    val_distr = torch.tensor([], requires_grad=False)
    val_targ = torch.tensor([], requires_grad=False)

    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        if isinstance(image, torch.Tensor):
            image = image.to(device, non_blocking=True)

        if isinstance(target, torch.Tensor):
            target = target.to(device, non_blocking=True)

        if fsim_enabled==True:
            output = model(image)
            Fsim_setup.FI_report.update_classification_report(im,output,target,topk=(1,5))
        else:
            output = model(image)

        cpu_target = target.to('cpu')
        val_targ = torch.cat((cpu_target, val_targ), dim = -1)

        soft = torch.nn.Softmax(dim=1)
        cpu_output = output.to('cpu')
        distr = soft(cpu_output)
        val_distr = torch.cat((distr, val_distr), dim = 0)

        acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
        # FIXME need to take into account that the datasets
        # could have been padded in distributed setup
        batch_size = len(image)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        im+=1

    if fsim_enabled==True:
        val_targ = val_targ.type(torch.int64)
        f1_1 = MulticlassF1Score(task='multiclass', num_classes=10, average='macro')
        rec_1 = MulticlassRecall(average='macro', num_classes=10)
        prec_1 = MulticlassPrecision(average='macro', num_classes=10)

        best_f1 = f1_1(val_distr, val_targ)
        best_rec = rec_1(val_distr, val_targ)
        best_prec = prec_1(val_distr, val_targ)

        f1_k = MulticlassF1Score(task='multiclass', num_classes=10, average='macro', top_k=5)
        rec_k = MulticlassRecall(num_classes=10, average='macro', top_k=5)
        prec_k = MulticlassPrecision(num_classes=10, average='macro', top_k=5)
        # logger.info(f'val_distr: {val_distr}')
        # logger.info(f'val_targ: {val_targ}')
        # logger.info(f'val_distr.shape: {val_distr.shape}')
        # logger.info(f'val_targ.shape: {val_targ.shape}')
        k_f1 = f1_k(val_distr, val_targ)
        k_rec = rec_k(val_distr, val_targ)
        k_prec = prec_k(val_distr, val_targ)

        Fsim_setup.FI_report.set_f1_values(best_f1=best_f1, k_f1=k_f1, header=header, best_prec= best_prec, best_rec = best_rec, k_prec= k_prec, k_rec = k_rec)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    if analyzable and model_wo_ddp.activated_analysis:
        model_wo_ddp.summarize()
    return metric_logger.acc1.global_avg

class LeNet5(torch.nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = torch.nn.Linear(400, 120)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(120, 84)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(84, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

def rec_dnn_exploration(model, new_layer=None):
    for idx in range(len(model._modules.values())):
        module = model._modules[list(model._modules.keys())[idx]]
        name = list(model._modules.keys())[idx]
        if isinstance(module, torch.nn.BatchNorm2d):
            # Check if the next layer is ReLU
            if idx != len(model._modules.values())-1:
                next_module = model._modules[list(model._modules.keys())[idx+1]]
                next_name = list(model._modules.keys())[idx+1]
                if isinstance(next_module, torch.nn.ReLU):
                    # Exchange positions of BatchNorm and ReLU
                    setattr(model, name, torch.nn.ReLU6(inplace=True))
                    setattr(model, next_name, module)

        else:
            # Recursively explore the next layer
            rec_dnn_exploration(module)
 
    return model




def main(args):
    log_file_path = args.log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    #distributed, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    distributed, device_ids = False, None
    logger.info(args)
    cudnn.enabled=True
    # cudnn.benchmark = True
    cudnn.deterministic = True
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    cudnn.allow_tf32 = True
    
    set_seed(args.seed)
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))
    if args.json is not None:
        logger.info('Overwriting config')
        overwrite_config(config, json.loads(args.json))

    device = torch.device(args.device)

    teacher_model = LeNet5(num_classes=10)
    if args.log_config:
        logger.info(config)
    
    models_config = config['models']
    
    teacher_model_config = models_config.get('teacher_model', None)

    # teacher_model = torch.load(
    #         os.path.join(teacher_model_config['ckpt']), map_location=torch.device("cpu")
    #     )
    teacher_model = LeNet5(num_classes=10)
    teacher_model.layer1[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(torch.tensor(6.4008), decimals=3), inplace=True)
    teacher_model.layer2[2]= torch.nn.Hardtanh(min_val=0, max_val=torch.round(torch.tensor(8.2039), decimals=3), inplace=True)
    teacher_model.relu= torch.nn.Hardtanh(min_val=0, max_val=torch.round(torch.tensor(27.2505), decimals=3), inplace=True)
    teacher_model.relu1= torch.nn.Hardtanh(min_val=0, max_val=torch.round(torch.tensor(45.0015), decimals=3), inplace=True)
    # teacher_model = rec_dnn_exploration(teacher_model)

    teacher_model.load_state_dict(torch.load(teacher_model_config['ckpt'])['state_dict'])

    logger.info(teacher_model)
    test_config = config['test']
    batch_size=1

    test_dataset = torchvision.datasets.MNIST(
        root='~/dataset',
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1325,), std=(0.3105,)),
            ]
        ),
        download=True,
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    models_config = config['models']
    


    log_freq = test_config.get('log_freq', 1000)
    no_dp_eval = args.no_dp_eval
        

    test_batch_size=config['test']['test_data_loader']['batch_size']
    test_shuffle=config['test']['test_data_loader']['random_sample']
    test_num_workers=config['test']['test_data_loader']['num_workers']
    subsampler = DatasetSampling(test_data_loader.dataset,10)
    index_dataset=subsampler.listindex()
    data_subset=Subset(test_data_loader.dataset, index_dataset)
    dataloader = DataLoader(data_subset,batch_size=test_batch_size, shuffle=test_shuffle,pin_memory=True,num_workers=test_num_workers)


    if args.fsim_config:
        fsim_config_descriptor = yaml_util.load_yaml_file(os.path.expanduser(args.fsim_config))
        conf_fault_dict=fsim_config_descriptor['fault_info']['weights']
        cwd=os.getcwd() 
        teacher_model.eval()
        # student_model.deactivate_analysis()
        # full_log_path=os.path.join(cwd,name_config)
        full_log_path=cwd
        # 1. create the fault injection setup
        FI_setup=FI_manager(full_log_path,"ckpt_FI.json","fsim_report.csv")

        # 2. Run a fault free scenario to generate the golden model
        FI_setup.open_golden_results("Golden_results")
        evaluate(teacher_model, dataloader, device, device_ids, distributed, no_dp_eval=no_dp_eval,
                log_freq=log_freq, title='[Teacher: {}]'.format(teacher_model_config['name']), header='Golden', fsim_enabled=True, Fsim_setup=FI_setup) 
        FI_setup.close_golden_results()
        trials = 50
        # 3. Prepare the Model for fault injections
        FI_setup.FI_framework.create_fault_injection_model(device,teacher_model,
                                            batch_size=1,
                                            input_shape=[1,32,32],
                                            layer_types=[torch.nn.Conv2d,torch.nn.Linear] )
        # input("wait for a second...")
        # 4. generate the fault list
        ber_list = list(conf_fault_dict['ber_list'])
        logging.getLogger('pytorchfi').disabled = True
        start = time.time()
        FI_setup.generate_fault_list(flist_mode='static_ber_fixed_layr',
                                     f_list_file='fault_list.csv',
                                     ber_list= ber_list,
                                     layr=int(args.layr_idx),
                                     trials = trials)
        end = time.time()
        logger.info(start-end)
        # FI_setup.load_check_point()
        
        # # 5. Execute the fault injection campaign
        counter = 0
        # logger.info(ber_list)
        for ber in ber_list:
            for trial in range(trials):
                fault_description =  FI_setup.get_trial_list(ber, trial)
                # 5.1 inject the fault in the model
                FI_setup.FI_framework.ber_var_bit_flip_weight_inj(fault_description, ber, trial)
                FI_setup.open_faulty_results(f"F_{ber}_{trial}_results")
                try:
                    # 5.2 run the inference with the faulty model 
                    evaluate(FI_setup.FI_framework.faulty_model, dataloader, device, device_ids, distributed, no_dp_eval=no_dp_eval,
                        log_freq=log_freq, title='[Teacher: {}]'.format(teacher_model_config['name']), header='FSIM', fsim_enabled=True,Fsim_setup=FI_setup)        
                except Exception as Error:
                    msg=f"Exception error: {Error}"
                    logger.info(msg)
                # 5.3 Report the results of the fault injection campaign            
                FI_setup.parse_results()
                counter += 1
        #     # break
        FI_setup.terminate_fsim()


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
