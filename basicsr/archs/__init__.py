# basicsr/archs/__init__.py
import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# List module names without importing them upfront
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
_arch_modules = [f'basicsr.archs.{file_name}' for file_name in arch_filenames]

def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net

def define_network(opt):
    network_type = opt.pop('type')
    net = dynamic_instantiation(_arch_modules, network_type, opt)
    return net

def dynamic_instantiation(modules, cls_type, opt):
    for module_name in modules:
        module = importlib.import_module(module_name)
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')
    return cls_(**opt)

# Register any explicitly imported architecture
def __getattr__(name):
    if name in [f.replace('_arch', '') for f in arch_filenames]:
        module_name = f'basicsr.archs.{name}_arch'
        module = importlib.import_module(module_name)
        cls_ = getattr(module, name, None)
        if cls_ is not None:
            return cls_
        raise AttributeError(f"'{name}' not found in {module_name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
