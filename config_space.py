import ConfigSpace as CS
import logging
from copy import deepcopy
logging.basicConfig(level=logging.ERROR)

def get_config_space():
    config_space = CS.ConfigurationSpace()

    learning_rate = CS.UniformFloatHyperparameter('learning_rate', lower=1e-6, upper=1e-1, default_value=1e-4, log=True)
    config_space.add_hyperparameter(learning_rate)

    batch_size = CS.UniformIntegerHyperparameter('batch_size', lower=8, default_value=32, upper=512, log=True)
    config_space.add_hyperparameter(batch_size)

    reg = CS.UniformFloatHyperparameter('reg', lower=1e-8, upper=1e-1, default_value=1e-4, log=True)
    config_space.add_hyperparameter(reg)

    return config_space