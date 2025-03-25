# basicsr/losses/__init__.py
import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import LOSS_REGISTRY

__all__ = ['build_loss']

# List module names without importing them
loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_loss.py')]
_loss_modules = [f'basicsr.losses.{file_name}' for file_name in loss_filenames]

def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Loss type.
    """
    opt = deepcopy(opt)
    loss = LOSS_REGISTRY.get(opt['type'])(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss

def dynamic_instantiation(modules, cls_type, opt):
    for module_name in modules:
        module = importlib.import_module(module_name)
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')
    return cls_(**opt)
