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
    return parser


def load_model(model_config, device, distributed):
    if 'classification_model' not in model_config:
        return load_classification_model(model_config, device, distributed)
    return get_wrapped_classification_model(model_config, device, distributed)



@torch.inference_mode()
def evaluate(model_wo_ddp, data_loader, device, device_ids, distributed, no_dp_eval=False,
             log_freq=1000, title=None, header='Test:', fsim_enabled=False, Fsim_setup:FI_manager = None):
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
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        if isinstance(image, torch.Tensor):
            image = image.to(device, non_blocking=True)

        if isinstance(target, torch.Tensor):
            target = target.to(device, non_blocking=True)

        if fsim_enabled==True:
            output = model(image)
            Fsim_setup.FI_report.update_report(im,output,target,topk=(1,5))
        else:
            output = model(image)

        acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
        # FIXME need to take into account that the datasets
        # could have been padded in distributed setup
        batch_size = len(image)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        im+=1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    if analyzable and model_wo_ddp.activated_analysis:
        model_wo_ddp.summarize()
    return metric_logger.acc1.global_avg




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
    dataset_dict = util.get_all_datasets(config['datasets'])
    models_config = config['models']
    teacher_model_config = models_config.get('teacher_model', None)
    teacher_model =\
        load_model(teacher_model_config, device, distributed) if teacher_model_config is not None else None

    if args.log_config:
        logger.info(config)


    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                              test_data_loader_config, distributed)
    log_freq = test_config.get('log_freq', 1000)
    no_dp_eval = args.no_dp_eval
        

    test_batch_size=config['test']['test_data_loader']['batch_size']
    test_shuffle=config['test']['test_data_loader']['random_sample']
    test_num_workers=config['test']['test_data_loader']['num_workers']
    subsampler = DatasetSampling(test_data_loader.dataset,5)
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

        # 3. Prepare the Model for fault injections
        FI_setup.FI_framework.create_fault_injection_model(device,teacher_model,
                                            batch_size=1,
                                            input_shape=[3,224,224],
                                            layer_types=[torch.nn.Conv2d,torch.nn.Linear])
        # input("wait for a second...")
        # 4. generate the fault list
        logging.getLogger('pytorchfi').disabled = True
        FI_setup.generate_fault_list(flist_mode='sbfm',f_list_file='fault_list.csv',layer=conf_fault_dict['layer'][0])    
        FI_setup.load_check_point()

        # 5. Execute the fault injection campaign
        for fault,k in FI_setup.iter_fault_list():
            # 5.1 inject the fault in the model
            FI_setup.FI_framework.bit_flip_weight_inj(fault)
            FI_setup.open_faulty_results(f"F_{k}_results")
            try:   
                # 5.2 run the inference with the faulty model 
                evaluate(FI_setup.FI_framework.faulty_model, dataloader, device, device_ids, distributed, no_dp_eval=no_dp_eval,
                    log_freq=log_freq, title='[Teacher: {}]'.format(teacher_model_config['name']), header='FSIM', fsim_enabled=True,Fsim_setup=FI_setup)        
            except Exception as Error:
                msg=f"Exception error: {Error}"
                logger.info(msg)
            # 5.3 Report the results of the fault injection campaign            
            FI_setup.parse_results()
            # break
        FI_setup.terminate_fsim()


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
