import time
import numpy as np
import random
from wgan_v3 import *

from hpbandster.distributed.worker import Worker

import logging

logging.basicConfig(level=logging.DEBUG)

def objective_function(config, epoch=127, **kwargs):
    # Cast the config to an array such that it can be forwarded to the surrogate

    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    reg = config['reg']
    #os.environ['CUDA_VISIBLE_DEVICES'] = 0
    data = importlib.import_module('mnist')
    model = importlib.import_module('mnist.dcgan')
    xs = data.DataSampler()
    zs = data.NoiseSampler()
    d_net = model.Discriminator()
    g_net = model.Generator()
    wgan = WassersteinGAN(g_net, d_net, xs, zs, data, model, l_rate=learning_rate,
                          batch_size=batch_size, reg=reg, epochs=epoch)
    wgan_list, g_loss_list = wgan.train()

    return wgan_list, g_loss_list

class MyWorker(Worker):
    def compute(self, config, budget, *args, **kwargs):
        loss, g_l = objective_function(config, epoch=int(budget))
        l = float(loss[-1])
        lc = [float(i) for i in loss]
        return ({
            'loss': l,
            'learning_curve':lc
        })