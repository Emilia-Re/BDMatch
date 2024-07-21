import argparse
import random

import numpy as np

from semilearn.core.utils import over_write_args_from_file
from train import get_config

if __name__=='__main__':

    #read config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, default='')
    args = parser.parse_args(args=['--c', 'config/classic_cv_os/bdmatch/bdmatch_cifar10_6_50_0.yaml'])
    over_write_args_from_file(args, args.c)

    #load dataset

    #load model
    _net_builder = get_net_builder(args.net, args.net_from_name)
        # optimizer, scheduler, datasets, dataloaders with be set in algorithms
    model = get_os_algorithm(args, _net_builder, tb_log, logger)
    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

        # SET Devices for (Distributed) DataParallel
    model.model = send_model_cuda(args, model.model)
    model.ema_model.model.cuda(args.gpu)
        # model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
    logger.info(f"Arguments: {model.args}")

        # If args.resume, load checkpoints from args.load_path

    model.load_model(args.load_path)



    #evalute
    logger.info("Model evaluating!")

    #get metrics

