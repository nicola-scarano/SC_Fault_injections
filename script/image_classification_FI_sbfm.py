import argparse
import datetime
import json
import os
import time
import sys

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

import multiprocessing as mp
from multiprocessing import TimeoutError
import subprocess
import psutil
import traceback
import gc
import requests
import sys


logger = def_logger.getChild(__name__)
# comment this line, otherwise the fault injections will collapse due to leaking memory produced by 'file_system
# torch.multiprocessing.set_sharing_strategy('file_system')
import logging



def send_to_telegram(message):

    apiToken = '6152839244:AAGam9_rwCHyo5OyLswcMuKreUfERMDbyjU'
    chatID = '665841436'
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'

    try:
        response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})
        print(response.text)
    except Exception as e:
        print(e)



class mpProcess(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


def cmdline(command):
    process = subprocess.Popen(args=command, stdout=subprocess.PIPE, shell=True)
    return process.communicate()[0]


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


def update_report_shared_dict(shared_dict_pred,shared_dict_clas,shared_dict_target,index,output,target,topk=(1,)):
    maxk=max(topk)        
    pred, clas=output.cpu().topk(maxk,1,True,True)
    shared_dict_pred[index]=pred.tolist()
    shared_dict_clas[index]=clas.tolist()
    shared_dict_target[index]=target.cpu().tolist()


@torch.inference_mode()
def evaluate(model_wo_ddp, data_loader, device, device_ids, distributed, 
            no_dp_eval, log_freq, title, header, fsim_enabled, Fsim_setup:FI_manager):
            #no_dp_eval=False, log_freq=1000, title=None, header='Test:', fsim_enabled=False, Fsim_setup = None):
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
            #image = image.to(device)

        if isinstance(target, torch.Tensor):
            target = target.to(device, non_blocking=True)
            #target = target.to(device)

        if fsim_enabled==True:
            output = model(image)

            #Fsim_setup.FI_report.update_report(im,output,target,topk=(1,5))
            Fsim_setup.FI_report.update_report_shared_dict(im,output,target,topk=(1,5))
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
        val_top1_accuracy = evaluate( student_model, training_box.val_data_loader, device, device_ids, distributed,
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
        
    test_batch_size=32#config['test']['test_data_loader']['batch_size']
    test_shuffle=config['test']['test_data_loader']['random_sample']
    test_num_workers=8#config['test']['test_data_loader']['num_workers']
    subsampler = DatasetSampling(test_data_loader.dataset,1)
    index_dataset=subsampler.listindex()
    data_subset=Subset(test_data_loader.dataset, index_dataset)
    dataloader = DataLoader(data_subset,batch_size=test_batch_size, shuffle=test_shuffle,pin_memory=True,num_workers=test_num_workers)

    #name_config=((args.config.split('/'))[-1]).replace(".yaml","")
    #conf_fault_dict=config['fault_info']['weights']
    #name_config=f"FSIM_logs/{name_config}_weights_{conf_fault_dict['layer'][0]}"
    if args.fsim_config:

        myhost = os.uname()[1]
        fsim_config_descriptor = yaml_util.load_yaml_file(os.path.expanduser(args.fsim_config))
        conf_fault_dict=fsim_config_descriptor['fault_info']['weights']
        cwd=os.getcwd() 
        send_to_telegram(f"{myhost}: WeightsFSIM: layer: {conf_fault_dict['layer'][0]} Simulation_started!..")
        # student_model.deactivate_analysis()
        #full_log_path=os.path.join(cwd,name_config)
        full_log_path=cwd
        # 1. create the fault injection setup
        FI_setup=FI_manager(full_log_path,"ckpt_FI.json","fsim_report.csv")

        # 2. Run a fault free scenario to generate the golden model

        FI_setup.open_golden_results("Golden_results")
        evaluate(student_model, dataloader, device, device_ids, distributed, 
                no_dp_eval, log_freq, '[Student: {}]'.format(student_model_config['name']), 'Golden', True, FI_setup) 
                #no_dp_eval=False, log_freq=1000, title=None, header='Test:', fsim_enabled=False, Fsim_setup = None):
        #FI_setup.FI_report._report_dictionary=FI_setup.FI_report._shared_dict.copy()
        FI_setup.FI_report.merge_shared_report() 
        FI_setup.close_golden_results()

        # 3. Prepare the Model for fault injections
        FI_setup.FI_framework.create_fault_injection_model(torch.device("cpu"),student_model.cpu(),
                                            batch_size=1,
                                            input_shape=[3,224,224],
                                            layer_types=[torch.nn.Conv2d,torch.nn.Linear])
        
        # 4. generate the fault list
        logging.getLogger('pytorchfi.core').disabled = True
        logging.getLogger('pytorchfi.FI_Weights').disabled = True
        logging.getLogger('pytorchfi.neuron_error_models').disabled = True
        
        FI_setup.generate_fault_list(flist_mode='sbfm',f_list_file='fault_list.csv',layer=conf_fault_dict['layer'][0])    
        FI_setup.load_check_point()
        top_PID = os.getpid()
        #signal.signal(signal.SIGALRM, handler)      
        print(os.getpid(),os.getppid())
        # 5. Execute the fault injection campaign
        
        gc.collect()
        torch.cuda.empty_cache()
        
        mp.set_start_method('spawn',force=True)
        #ctx = mp.get_context('spawn')
        #q = ctx.Queue()
        with mp.Manager() as manager:

            for fault,k in FI_setup.iter_fault_list():
                logger.info(f"index {k}: start")
                #GPUResMemAlloc = cmdline("lsof /dev/nvidia0")
                #GPUResMemAlloc = str(GPUResMemAlloc).replace('\\n', '\n')
                #print(GPUResMemAlloc)
                gc.collect()
                with torch.no_grad():
                    torch.cuda.empty_cache()

                #print(torch.cuda.memory_summary())
                # 5.1 inject the fault in the model
                FI_setup.FI_report.shared_dict_pred=manager.dict()
                FI_setup.FI_report.shared_dict_clas=manager.dict()
                FI_setup.FI_report.shared_dict_target=manager.dict()
                ############## input("XXXXXX")
                FI_setup.FI_framework.bit_flip_weight_inj(fault)
                FI_setup.open_faulty_results(f"F_{k}_results") 

                #signal.alarm(30)  
                # 5.2 run the inference with the faulty model          
                Process = mpProcess(target=evaluate, args=(FI_setup.FI_framework.faulty_model, dataloader, device, device_ids, distributed, no_dp_eval,
                    log_freq, '[Student: {}]'.format(student_model_config['name']), 'FSIM', True,FI_setup,))          
                #Process = ctx.Process(target=evaluate, args=(FI_setup.FI_framework.faulty_model, dataloader, device, device_ids, distributed, no_dp_eval,
                #    log_freq, '[Student: {}]'.format(student_model_config['name']), 'FSIM', True,FI_setup,))  
                #Process = torch.multiprocessing.spawn(fn=evaluate, args=(FI_setup.FI_framework.faulty_model, dataloader, device, device_ids, distributed, no_dp_eval,
                #    log_freq, '[Student: {}]'.format(student_model_config['name']), 'FSIM', True,FI_setup,),nprocs=)          
                
                
                try:                       
                    Process.start()
                    Process.join(timeout=60)                    
                    #print(shared_dict_pred,shared_dict_clas,shared_dict_target)                    
                    if Process.is_alive():
                        raise TimeoutError
                    
                    FI_setup.FI_report.merge_shared_report()
                        #FI_setup.FI_framework.faulty_model.__del__()
                        #print(FI_setup.FI_report._shared_dict)
                    # evaluate(FI_setup.FI_framework.faulty_model, dataloader, device, device_ids, distributed, no_dp_eval=no_dp_eval,
                    #     log_freq=log_freq, title='[Student: {}]'.format(student_model_config['name']), header='FSIM', fsim_enabled=True,Fsim_setup=FI_setup)                     
                    
                except TimeoutError:
                    FI_setup.FI_report.merge_shared_report()
                    logger.info(f"Exception error: Timeout")   
                    logger.info(f"index {k}: The DNN inference got stuck")   
                    #del FI_setup.FI_framework.pfi_model.corrupted_model
                    while Process.is_alive():
                        Process.terminate()                            
                    Process.close()

                    #FI_setup.FI_framework.faulty_model.__del__()
                    
                except Exception as gpuerr:
                    msg=f"Exception error: {gpuerr}"
                    logger.info(msg)
                    logger.info(f"index {k}: an unexpected error happened during inference")  
                    FI_setup.FI_report.merge_shared_report() 
                    if Process.is_alive():                        
                        while Process.is_alive():
                            Process.terminate()
                        Process.close()
                        #Process.join()                      
                # 5.3 Report the results of the fault injection campaign       
                if Process.exception:
                    error, traceback = Process.exception
                    if "out of memory" in str(error) or "out of memory" in str(traceback):
                        msg=f"Exception error: {error}"
                        logger.info(msg)
                        send_to_telegram(f"{myhost}: WeightsFSIM: {msg}") 
                        sys.exit(0)

                FI_setup.parse_results()
                # p_children =[]
                # for child in ProcessHier.children(recursive=True):
                #     p_children.append((child.pid,child.name()))      
                # 
                # for (c_pid,c_name) in p_children:
                #     print(c_pid,c_name)
            FI_setup.terminate_fsim()
            
        """
        for fault,k in FI_setup.iter_fault_list():
            logger.info(f"index {k}: start")
            # 5.1 inject the fault in the modelProcess.pid
            FI_setup.FI_framework.bit_flip_weight_inj(fault)
            logger.info(f"index {k}: fault injected succesfully")
            FI_setup.open_faulty_results(f"F_{k}_results") 
            logger.info(f"index {k}: faulty report created succesfully") 
              
            signal.alarm(30)  
            try:   
                # 5.2 run the inference with the faulty model 
                logger.info(f"index {k}: inference started")                              
                evaluate(FI_setup.FI_framework.faulty_model, dataloader, device, device_ids, distributed, no_dp_eval=no_dp_eval,
                    log_freq=log_freq, title='[Student: {}]'.format(student_model_config['name']), header='FSIM', fsim_enabled=True,Fsim_setup=FI_setup) 
                logger.info(f"index {k}: Inference completed")

            except TimeOutException as exc:
                msg=f"Exception error: {exc}"
                logger.info(msg)
                logger.info(f"index {k}: an Timeout")                            
            except Exception as Error:
                msg=f"Exception error: {Error}"
                logger.info(msg)
                logger.info(f"index {k}: an unexpected error happened during inference")            

            # 5.3 Report the results of the fault injection campaign            
            FI_setup.parse_results()
        FI_setup.terminate_fsim()
        """
if __name__ == '__main__':

    argparser = get_argparser()
    main(argparser.parse_args())
