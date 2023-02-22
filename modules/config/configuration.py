import copy
import logging
import os
from pathlib import Path
from typing import Dict, Union, Any

from modules.config.default_config import DEFAULT_CONFIGS, arch_default_config, LOG_LEVELS
from modules.config.config_utils import convert_config_dict, load_cmd_line, setup_project_path
from modules.utils import write_json, read_json


class Configuration:
    """
    This is the base class for all configuration class. Deal with the common hyper-parameters to all models'
    configuration, and include the methods for loading/saving configurations.
    Configuration module allows the above three kind of external input format to be used together,
    the priority order is as following:

    command line > parameter dictionaries > config file > default configuration
    """

    def __init__(self, **kwargs):
        self.final_configs = DEFAULT_CONFIGS  # load default configuration first
        config_file = kwargs.get("config_file", None)
        if config_file is not None and os.path.exists(config_file):
            self.update(convert_config_dict(read_json(config_file)))  # load config file second
        self.update(kwargs.get("config_dict", {}))  # load parameter dictionary third
        self.update(load_cmd_line())  # load command line parameters finally
        self.final_configs = setup_project_path(self.final_configs)
        self.save_config(self.model_dir)  # save updated config file to the checkpoint directory

    def __getattr__(self, item):
        if "final_configs" not in self.__dict__:
            raise AttributeError(f"'Config' object has no attribute 'final_config_dict'")
        if item in self.final_configs:
            return self.final_configs[item]
        for k, v in self.final_configs.items():
            if isinstance(v, dict) and item in v:
                return v[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __hasattr__(self, item):
        if "final_configs" not in self.__dict__:
            return False
        if item in self.final_configs:
            return True
        for k, v in self.final_configs.items():
            if isinstance(v, dict) and item in v:
                return True
        return False

    def __getitem__(self, item):
        """Access items like ordinary dict."""
        return self.get(item)

    def __setitem__(self, key, value):
        self.set(key, value)

    def __str__(self):
        # print the configuration
        args_info = "\n"
        for k, v in self.final_configs.items():
            args_info += f"{k}: {v}\n"
        return args_info

    def get_logger(self, name, verbosity=2):
        msg_verbosity = f"verbosity option{verbosity} is invalid. Valid options are {LOG_LEVELS.keys()}."
        assert verbosity in LOG_LEVELS, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(LOG_LEVELS[verbosity])
        return logger

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return self.final_configs.get(key, default)

    def set(self, key, value):
        self.final_configs[key] = value

    def update(self, config_dict: Dict[str, Any]):
        """
        Updates attributes of this class with attributes from ``config_dict``.
        Also update architecture default parameters too.

        Args:
            config_dict (:obj:`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        if "arch_type" in config_dict:
            self.final_configs.update(arch_default_config(config_dict["arch_type"]))
        self.final_configs.update(config_dict)

    def save_config(self, save_dir: Union[str, os.PathLike], config_name: str = "config.json"):
        """
        Save configuration with the saved directory with corresponding configuration name in a json file
        :param config_name: default is config.json, should be a json filename
        :param save_dir: the directory to save the configuration
        """
        if os.path.isfile(save_dir):
            raise AssertionError(f"Provided path ({save_dir}) should be a directory, not a file")
        os.makedirs(save_dir, exist_ok=True)
        config_file = Path(save_dir) / config_name
        write_json(copy.deepcopy(self.final_configs), config_file)


if __name__ == "__main__":
    config = Configuration()
