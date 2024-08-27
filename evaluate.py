
import argparse
import os
import logging
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from semilearn.os_algorithms import get_os_algorithm, name2osalg
from semilearn.core.utils import get_net_builder, get_logger, get_port, send_model_cuda, count_parameters, over_write_args_from_file, TBLog
from train import get_config

if __name__=='__main__':


    #read config file
    args = get_config()
    args.c=r'config/classic_cv_os/bdmatch/bdmatch_cifar10_6_4_0_test.yaml'
    over_write_args_from_file(args, args.c)
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node
    '''
    main_worker is conducted on each GPU.
    '''

    global best_acc1
    args.gpu = None

    # random seed has to be set for the synchronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    ngpus_per_node=torch.cuda.device_count()

    args.gpu = 0

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    logger.info(f"Use GPU: {args.gpu} for training")

    _net_builder = get_net_builder(args.net, args.net_from_name)
    # optimizer, scheduler, datasets, dataloaders with be set in algorithms
    model = get_os_algorithm(args, _net_builder, tb_log, logger)  # 'model' is a BDmatch class
    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    # SET Devices for (Distributed) DataParallel
    model.model = send_model_cuda(args, model.model)
    model.ema_model.model.cuda(args.gpu)
    # model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
    logger.info(f"Arguments: {model.args}")

    # If args.resume, load checkpoints from args.load_path
    if args.resume and os.path.exists(args.load_path):
        try:
            model.load_model(args.load_path)
        except:
            logger.info("Fail to resume load path {}".format(args.load_path))
            args.resume = False
    else:
        logger.info("Resume load path {} does not exist".format(args.load_path))

    if hasattr(model, 'warmup'):
        logger.info(("Warmup stage"))
        model.warmup()

    logger.info("Evaluating!")
    # model.call_hook('OSEvaluationHook')

    model.hooks_dict['OSEvaluationHook'].evaluate(model)# 从这里进入evulate 部分的代码,实际运行的hook有OSEvaluationHook
    # print validation (and test results)
    for key, item in model.results_dict.items():
        logger.info(f"Model result - {key} : {item}")

    if hasattr(model, 'finetune'):
        logger.info("Finetune stage")
        model.finetune()

    # logging.warning(f"GPU {args.rank} training is FINISHED")


