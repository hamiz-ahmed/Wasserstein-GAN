from wgan import *
from hpbandster.distributed.worker import Worker
import logging

logging.basicConfig(level=logging.DEBUG)


def objective_function(config, epoch=127, **kwargs):
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    data = importlib.import_module('mnist')
    model = importlib.import_module('mnist.dcgan')
    xs = data.DataSampler()
    zs = data.NoiseSampler()
    d_net = model.Discriminator()
    g_net = model.Generator()
    wgan = WassersteinGAN(g_net, d_net, xs, zs, data, model, l_rate=learning_rate,
                          batch_size=batch_size, iterations=epoch)
    w_dist_list, g_loss_list = wgan.train()

    return w_dist_list, g_loss_list

class MyWorker(Worker):
    """
    Worker for Hpbandster in Hyperparameter optimization
    """

    def compute(self, config, budget, *args, **kwargs):
        w_dist_l, g_l = objective_function(config, epoch=int(budget))
        l = float(w_dist_l[-1])
        lc = [float(i) for i in w_dist_l]
        return ({
            'loss': l,
            'learning_curve':lc
        })