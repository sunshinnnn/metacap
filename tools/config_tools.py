"""
@Author: Guoxing Sun
@Email: gsun@mpi-inf.mpg.de
@Date: 2023-01-23
"""
from dotmap import DotMap
import os
import yaml

def load_config_with_default(default_path, path, log=True, logger=None):
    cfgs = load_config(default_path)
    cfgExp = load_config(path)
    cfgs = cfgs.toDict()
    cfgExp = cfgExp.toDict()
    for key in cfgExp.keys():
        cfgs[key] = cfgExp[key]
    if log:
        print_config(cfgs, logger=logger)
    return DotMap(cfgs, _dynamic=False)

def load_config(default_ps_fname=None, **kwargs):
    if isinstance(default_ps_fname, str):
        assert os.path.exists(default_ps_fname), FileNotFoundError(default_ps_fname)
        assert default_ps_fname.lower().endswith('.yaml'), NotImplementedError('Only .yaml files are accepted.')
        default_ps = yaml.safe_load(open(default_ps_fname, 'r'))
    else:
        default_ps = {}

    default_ps.update(kwargs)

    return DotMap(default_ps, _dynamic=False)

def dump_config(data, fname):
    '''
    dump current configuration to an ini file
    :param fname:
    :return:
    '''
    with open(fname, 'w') as file:
        yaml.dump(data.toDict(), file)
    return fname

def print_config(cfg, logger=None):
    message = ''
    message += '----------------- Configs ---------------\n'
    if isinstance( cfg , DotMap ):
        cfg = cfg.toDict()
    # for k, v in sorted(vars(cfg).items()):
    # for k, v in sorted(cfg.items()):
    for k, v in cfg.items():
        comment = ''
        if isinstance(v, dict):
            message += '{:->25}-----------------\n'.format(str(k))
            for ks, vs in v.items():
                message += '{:>25}: {:<30}{}\n'.format(str(ks), str(vs), comment)
            message += '{:->25}-----------------\n'.format(str(k))
        else:
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    if not logger is None:
        logger.info(message)
    else:
        print(message)