import os
import sys
import importlib
import random
import torch
import numpy as np

CLASS_NAME_MAP = {
    't_test_user':'TR_VALUENET'
}


def add_module_search_paths(search_paths: list) -> None:
    """Adds paths for searching python modules.
    """
    augmented_search_paths = search_paths
    for path in search_paths:
        for root, dirs, _ in os.walk(path):
            cur_dir_paths = [os.path.join(root, cur_dir) for cur_dir in dirs]
            augmented_search_paths.extend(cur_dir_paths)
    sys.path.extend(augmented_search_paths)

def get_class(module_name):
    class_name = CLASS_NAME_MAP[module_name]
    module = importlib.import_module(module_name)
    Class = getattr(module, class_name)
    return Class

def get_model(args, tokenizer):
    add_module_search_paths(['./model'])
    Model = get_class(args.model)
    model = Model(args, tokenizer)
    return model

def get_dataset(args, tokenizer, group_num, data_type):
    add_module_search_paths(['./dataset'])
    Dataset = get_class(args.dataset)
    dataset = Dataset(args, tokenizer, group_num, data_type)
    return dataset

def get_trainer(args, tokenizer, model, dataset, group_num, logger):
    add_module_search_paths(['./trainer'])
    Trainer = get_class(args.trainer)
    trainer = Trainer(args, tokenizer, model, dataset, group_num, logger)
    return trainer

def set_seed(seed):
    """Fixes randomness to enable reproducibility.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False