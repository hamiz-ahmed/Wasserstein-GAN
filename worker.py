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

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    data = importlib.import_module('mnist')
    model = importlib.import_module('dcgan')
    xs = data.DataSampler()
    zs = data.NoiseSampler()
    d_net = model.Discriminator()
    g_net = model.Generator()
    wgan = WassersteinGAN(g_net, d_net, xs, zs, args.data, args.model, l_rate=learning_rate,
                          batch_size=batch_size, reg=reg, epochs=epoch)
    wgan_list, g_loss_list = wgan.train()

    #x = deepcopy(config.get_array())
    #x[np.isnan(x)] = -1
    #lc = rf.predict(x[None, :])[0]
    #c = cost_rf.predict(x[None, :])[0]



    return lc[epoch], {"cost": c, "learning_curve": lc[:epoch].tolist()}

class MyWorker(Worker):

    def compute(self, config, budget, *args, **kwargs):
        """
            Simple example for a compute function

            The loss is just a the config + some noise (that decreases with the budget)
            There is a 10 percent failure probability for any run, just to demonstrate
            the robustness of Hyperband agains these []kinds of failures.

            For dramatization, the function sleeps for one second, which emphasizes
            the speed ups achievable with parallel workers.
        """

        time.sleep(1)

        # simulate some random failure
        if random.random() < 0.2:
            raise RuntimeError("Random runtime error!")

        res = []
        for i in range(int(budget)):
            tmp = config['x'] + np.random.randn() / budget
            res.append(tmp)

        return ({
            'loss': np.mean(res),  # this is the a mandatory field to run hyperband
            'info': res  # can be used for any user-defined information - also mandatory
        })