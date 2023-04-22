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

from pytorchfi.FI_Weights import FI_report_classifier
from pytorchfi.FI_Weights import FI_framework
from pytorchfi.FI_Weights import FI_manager 
from pytorchfi.FI_Weights import DatasetSampling 

from torch.utils.data import DataLoader, Subset

logger = def_logger.getChild(__name__)
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
    return parser


def load_model(model_config, device, distributed):
    if 'classification_model' not in model_config:
        return load_classification_model(model_config, device, distributed)
    return get_wrapped_classification_model(model_config, device, distributed)


def train_one_epoch(training_box, aux_module, bottleneck_updated, device, epoch, log_freq):
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', SmoothedValue(window_size=10, fmt='{value}'))
    uses_aux_loss = aux_module is not None and not bottleneck_updated
    header = 'Epoch: [{}]'.format(epoch)
    for sample_batch, targets, supp_dict in \
            metric_logger.log_every(training_box.train_data_loader, log_freq, header):
        if isinstance(sample_batch, torch.Tensor):
            sample_batch = sample_batch.to(device)

        if isinstance(targets, torch.Tensor):
            targets = targets.to(device)

        start_time = time.time()
        loss = training_box(sample_batch, targets, supp_dict)
        aux_loss = None
        if uses_aux_loss:
            aux_loss = aux_module.aux_loss()
            aux_loss.backward()

        training_box.update_params(loss)
        batch_size = len(sample_batch)
        if uses_aux_loss:
            metric_logger.update(loss=loss.item(), aux_loss=aux_loss.item(),
                                 lr=training_box.optimizer.param_groups[0]['lr'])
        else:
            metric_logger.update(loss=loss.item(), lr=training_box.optimizer.param_groups[0]['lr'])
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
        if (torch.isnan(loss) or torch.isinf(loss)) and is_main_process():
            raise ValueError('The training loop was broken due to loss = {}'.format(loss))


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


def train(teacher_model, student_model, dataset_dict, ckpt_file_path, device, device_ids, distributed, config, args):
    logger.info('Start training')
    train_config = config['train']
    lr_factor = args.world_size if distributed and args.adjust_lr else 1
    training_box = get_training_box(student_model, dataset_dict, train_config,
                                    device, device_ids, distributed, lr_factor) if teacher_model is None \
        else get_distillation_box(teacher_model, student_model, dataset_dict, train_config,
                                  device, device_ids, distributed, lr_factor)
    best_val_top1_accuracy = 0.0
    optimizer, lr_scheduler = training_box.optimizer, training_box.lr_scheduler
    if file_util.check_if_exists(ckpt_file_path):
        best_val_top1_accuracy, _, _ = load_ckpt(ckpt_file_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    log_freq = train_config['log_freq']
    student_model_without_ddp = student_model.module if module_util.check_if_wrapped(student_model) else student_model
    aux_module = student_model_without_ddp.get_aux_module() if check_if_updatable(student_model_without_ddp) else None
    epoch_to_update = train_config.get('epoch_to_update', None)
    bottleneck_updated = False
    no_dp_eval = args.no_dp_eval
    start_time = time.time()
    for epoch in range(args.start_epoch, training_box.num_epochs):
        training_box.pre_process(epoch=epoch)
        if epoch_to_update is not None and epoch_to_update <= epoch and not bottleneck_updated:
            logger.info('Updating entropy bottleneck')
            student_model_without_ddp.update()
            bottleneck_updated = True

        train_one_epoch(training_box, aux_module, bottleneck_updated, device, epoch, log_freq)
        val_top1_accuracy = evaluate(student_model, training_box.val_data_loader, device, device_ids, distributed,
                                     no_dp_eval=no_dp_eval, log_freq=log_freq, header='Validation:')
        if val_top1_accuracy > best_val_top1_accuracy and is_main_process():
            logger.info('Best top-1 accuracy: {:.4f} -> {:.4f}'.format(best_val_top1_accuracy, val_top1_accuracy))
            logger.info('Updating ckpt at {}'.format(ckpt_file_path))
            best_val_top1_accuracy = val_top1_accuracy
            save_ckpt(student_model_without_ddp, optimizer, lr_scheduler,
                      best_val_top1_accuracy, config, args, ckpt_file_path)
        training_box.post_process()

    if distributed:
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    training_box.clean_modules()


def main(args):
    log_file_path = args.log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    # distributed, device_ids = init_distributed_mode(args.world_size, args.dist_url)
    distributed, device_ids = False, None
    logger.info(args)
    cudnn.enabled=True
    # cudnn.benchmark = True
    cudnn.deterministic = True
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
    student_model_config =\
        models_config['student_model'] if 'student_model' in models_config else models_config['model']
    ckpt_file_path = student_model_config.get('ckpt', None)
    student_model = load_model(student_model_config, device, distributed)
    if args.log_config:
        logger.info(config)

    if not args.test_only:
        train(teacher_model, student_model, dataset_dict, ckpt_file_path, device, device_ids, distributed, config, args)
        student_model_without_ddp =\
            student_model.module if module_util.check_if_wrapped(student_model) else student_model
        load_ckpt(student_model_config['ckpt'], model=student_model_without_ddp, strict=True)

    test_config = config['test']
    test_data_loader_config = test_config['test_data_loader']
    test_data_loader = util.build_data_loader(dataset_dict[test_data_loader_config['dataset_id']],
                                              test_data_loader_config, distributed)
    log_freq = test_config.get('log_freq', 1000)
    no_dp_eval = args.no_dp_eval
    if not args.student_only and teacher_model is not None:
        evaluate(teacher_model, test_data_loader, device, device_ids, distributed, no_dp_eval=no_dp_eval,
                 log_freq=log_freq, title='[Teacher: {}]'.format(teacher_model_config['name']))

    

    if check_if_updatable(student_model):
        student_model.update()
        

    if check_if_analyzable(student_model):
        student_model.activate_analysis()
        

    test_batch_size=config['test']['test_data_loader']['batch_size']
    test_shuffle=config['test']['test_data_loader']['random_sample']
    test_num_workers=config['test']['test_data_loader']['num_workers']
    subsampler = DatasetSampling(test_data_loader.dataset,5)
    index_dataset=subsampler.listindex()
    data_subset=Subset(test_data_loader.dataset, index_dataset)
    dataloader = DataLoader(data_subset,batch_size=test_batch_size, shuffle=test_shuffle,pin_memory=True,num_workers=test_num_workers)

    name_config=((args.config.split('/'))[-1]).replace(".yaml","")
    conf_fault_dict=config['fault_info']['neurons']
    print(conf_fault_dict)
    name_config=f"FSIM_logs/{name_config}_neurons_{conf_fault_dict['layers'][0]}"


    cwd=os.getcwd() 
    teacher_model.eval() 
    # student_model.deactivate_analysis()
    full_log_path=os.path.join(cwd,name_config)
    # 1. create the fault injection setup
    FI_setup=FI_manager(full_log_path,chpt_file_name='ckpt_FI.json',fault_report_name='fsim_report.csv')

    # 2. Run a fault free scenario to generate the golden model
    FI_setup.open_golden_results("Golden_results")
    evaluate(teacher_model, dataloader, device, device_ids, distributed, no_dp_eval=no_dp_eval,
            log_freq=log_freq, title='[Student: {}]'.format(teacher_model_config['name']), header='Golden', fsim_enabled=True, Fsim_setup=FI_setup) 
    FI_setup.close_golden_results()

    # 3. Prepare the Model for fault injections
    FI_setup.FI_framework.create_fault_injection_model(device,teacher_model,
                                        batch_size=test_batch_size,
                                        input_shape=[3,224,224],
                                        layer_types=[torch.nn.Conv2d],Neurons=True)
    
    # 4. generate the fault list
    logging.getLogger('pytorchfi').disabled = True
    #logging.getLogger('pytorchfi.neuron_error_models').disabled = True
    FI_setup.generate_fault_list(flist_mode='neurons',
                                 f_list_file='fault_list.csv',
                                 layers=conf_fault_dict['layers'],
                                 trials=conf_fault_dict['trials'], 
                                 size_tail_y=conf_fault_dict['size_tail_y'], 
                                 size_tail_x=conf_fault_dict['size_tail_x'],
                                 block_fault_rate_delta=conf_fault_dict['block_fault_rate_delta'],
                                 block_fault_rate_steps=conf_fault_dict['block_fault_rate_steps'],
                                 neuron_fault_rate_delta=conf_fault_dict['neuron_fault_rate_delta'],
                                 neuron_fault_rate_steps=conf_fault_dict['neuron_fault_rate_steps'])     
      
    FI_setup.load_check_point()


    ber=5
    # 5. Execute the fault injection campaign
    for fault,k in FI_setup.iter_fault_list():
        # 5.1 inject the fault in the model
        #FI_setup.FI_framework.bit_flip_weight_inj([fault[0]],[fault[1]],[fault[2]],[fault[3]],[fault[4]],[fault[5]])
        FI_setup.FI_framework.bit_flip_err_neuron(fault)
        FI_setup.open_faulty_results(f"F_{k}_results")

        try:   
            # 5.2 run the inference with the faulty model 
            evaluate(FI_setup.FI_framework.faulty_model, dataloader, device, device_ids, distributed, no_dp_eval=no_dp_eval,
                log_freq=log_freq, title='[Student: {}]'.format(teacher_model_config['name']), header='FSIM', fsim_enabled=True,Fsim_setup=FI_setup)        
        
        except OSError as Oserr:
            msg=f"Oserror: {Oserr}"
            logger.info(msg)

        except Exception as Error:
            msg=f"Exception error: {Error}"
            logger.info(msg)
        
        # 5.3 Report the results of the fault injection campaign
        FI_setup.close_faulty_results()
        FI_setup.parse_results()
        FI_setup.write_reports()


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
