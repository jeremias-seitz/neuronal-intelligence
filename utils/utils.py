from omegaconf import OmegaConf, dictconfig
import torch
import numpy as np
import random


__all__ = ['generate_run_name_from_hydra', 
           'remove_target_key',
           'get_device_from_config', 
           'set_seed_all',
           'save_model']


def generate_run_name_from_hydra(config):
    """
    Example function to generate a run name from the hydra configuration. This can be useful to differentiate runs
    later on when logging (if e.g. the wandb service is used, see https://wandb.ai/home.)

    Args:
        config (dictconfig.DictConfig): Hydra configuration file

    Returns:
        string: wandb run name
    """

    if 'run_name_string' in config.keys():
        run_name = config.run_name_string
    else:
        run_name = config.loss_entity.name
        run_name += f"_C{config.loss_entity.regularization_strength}"
        if config.is_joint_training: run_name += "_joint"
        run_name += f"_{config.learning_scenario.name}"
        if config.optimizer.weight_decay > 0:
            run_name += f"_wdecay{config.optimizer.weight_decay:.0e}"

    return run_name


def remove_target_key(config: dictconfig.DictConfig) -> dictconfig.DictConfig:
    """
    Removes all '_target_' keys from the given config file. Hydra attempts to instantiate all objects in the 
    config when the config is passed as an argument. This is prevented when all the '_target_' keys are removed.

    Args:
        config (dictconfig.DictConfig): Hydra configuration file

    Returns:
        dictconfig.DictConfig: Hydra configuration file without '_target_' keys
    """
    dict_config = OmegaConf.to_container(config)
    remove_key_recursively(dictionary=dict_config, remove_key="_target_")
    return OmegaConf.create(dict_config)


def remove_key_recursively(dictionary: dict, remove_key: str):
    """
    Recursive removal of a specified key from a dictionary.

    Args:
        dictionary (dict): Dictionary where a key shall be removed
        remove_key (str): Key to be removed from all levels of the dictionary
    """
    if remove_key in dictionary.keys():
        dictionary.pop(remove_key)
    for value in dictionary.values():
        if isinstance(value, dict):
            remove_key_recursively(dictionary=value, remove_key=remove_key)


def get_device_from_config(config: dictconfig.DictConfig) -> str:
    """
    Generates the device string (either 'cpu' or 'cuda:<GPUID>') depending on the configuration.
    This is required for torch to know where to allocate the tensors for accelerated computation.

    Args:
        config (dictconfig.DictConfig): Hydra configuration file

    Returns:
        str: Device string (either 'cpu' or 'cuda:<GPUID>')
    """
    cuda_num = config.cuda_num
    
    if isinstance(cuda_num, int):
        device = f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    return device


def set_seed_all(seed:int):
    """
    Make the training process as deterministic as possible. Note that this makes training a lot slower.
    The main slowdown comes from the deterministic flag in the torch backend.

    Args:
        seed (int): Seed for the RNG
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)

        
def save_model(model: torch.nn.Module, path: str, filename: str):
    """
    Example function to save the model data.

    Args:
        model (torch.nn.Module): Model to be saved
        path (str): Path to the directory
        filename (str): Filename
    """    
    torch.save({
        'model_state_dict': model.state_dict(),
        }, f"{path}{filename}.pt")
    
    print(f"model saved: {filename}")
