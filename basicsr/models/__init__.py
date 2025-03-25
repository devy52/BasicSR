# basicsr/models/__init__.py
import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import MODEL_REGISTRY

__all__ = ['build_model']
__all__ = ['build_model']
__all__ = ['build_model']
__all__ = ['build_model']

# List module names without importing them
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
_model_modules = [f'basicsr.models.{file_name}' for file_name in model_filenames]

def build_model(opt):
    opt = deepcopy(opt)
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model

def dynamic_instantiation(modules, cls_type, opt):
    for module_name in modules:
        module = importlib.import_module(module_name)
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f'{cls_type} is not found.')
    return cls_(**opt)

# Register any explicitly imported model
def __getattr__(name):
    if name in [f.replace('_model', '') for f in model_filenames]:
        module_name = f'basicsr.models.{name}_model'
        module = importlib.import_module(module_name)
        cls_ = getattr(module, name, None)
        if cls_ is not None:
            return cls_
        raise AttributeError(f"'{name}' not found in {module_name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
