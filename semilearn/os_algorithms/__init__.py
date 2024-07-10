# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from semilearn.core.utils import OS_ALGORITHMS
name2osalg = OS_ALGORITHMS

def get_os_algorithm(args, net_builder, tb_log, logger):
    if args.os_algorithm in OS_ALGORITHMS:
        alg = OS_ALGORITHMS[args.os_algorithm](
            args=args,
            net_builder=net_builder,
            tb_log=tb_log,
            logger=logger
        )
        return alg
    else:
        raise KeyError(f'Unknown algorithm: {str(args.os_algorithm)}')



