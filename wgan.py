import os
import time
import argparse
import importlib
import tensorflow as tf
from scipy.misc import imsave
from visualize import *
from scipy import signal
import pickle


class WassersteinGAN(object):
    def __init__(self, g_net, d_net, x_sampler, z_sampler, data, model, iterations=1, l_rate=1e-4, batch_size=64):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.iterations = iterations
        self.l_rate = l_rate
        self.batch_size = batch_size
        self.model = model
        self.data = data
        self.g_net = g_net
        self.d_net = d_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.x_ = self.g_net(self.z)

        self.d = self.d_net(self.x, reuse=False)
        self.d_ = self.d_net(self.x_)

        self.g_loss = tf.reduce_mean(self.d_)
        self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)

        # self.d_loss = self.d_loss
        # self.g_loss = self.g_loss

        self.d_rms_prop, self.g_rms_prop = None, None

        # apply RMS Prop
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_rms_prop = tf.train.RMSPropOptimizer(learning_rate=self.l_rate)\
                .minimize(self.d_loss, var_list=self.d_net.vars)
            self.g_rms_prop = tf.train.RMSPropOptimizer(learning_rate=self.l_rate)\
                .minimize(self.g_loss, var_list=self.g_net.vars)

        # clip critic weights
        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_net.vars]
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self):
        dir = "logs/mnist/{}_{}".format(self.l_rate, self.batch_size)
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.ion()

        batch_size = self.batch_size
        iterations = self.iterations                    #1000000

        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()

        d_loss_list = []
        g_loss_list = []
        print('Number of Iterations: {}'.format(iterations))

        for t in range(0, iterations):
            d_iters = 5
            if t % 500 == 0 or t < 25:
                 d_iters = 25

            for _ in range(0, d_iters):
                bx = self.x_sampler(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim)
                self.sess.run(self.d_rms_prop, feed_dict={self.x: bx, self.z: bz})
                self.sess.run(self.d_clip)

            bz = self.z_sampler(batch_size, self.z_dim)
            self.sess.run(self.g_rms_prop, feed_dict={self.z: bz, self.x: bx})

            if t % 100 == 0:
                bx = self.x_sampler(batch_size)
                bz = self.z_sampler(batch_size, self.z_dim)

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz}
                )

                # append loss with a negative sign to attain same plots as in paper
                d_loss_list.append(-d_loss)
                g_loss_list.append(g_loss)

                #save readings in a text file
                with open('logs/w_dist/reading.txt', 'a+') as txt_file:
                    txt_file.write(str(-d_loss) + '\n')

                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                        (t, time.time() - start_time, -d_loss, g_loss))

            if t % 1000 == 0:
                # generate image after 1000 iterations
                self.generate_image(dir, t)

        # apply median filter to attain plots of the paper
        med_filtered_w_dist = signal.medfilt(d_loss_list)

        if not os.path.exists("logs/res"):
            os.makedirs("logs/res")

        # pickle generated lists for later use
        with open("logs/res/d_loss", "wb+") as d_loss_file:
            pickle.dump(d_loss_list, d_loss_file)

        with open("logs/res/median_d_loss", "wb+") as median_d_loss_file:
            pickle.dump(med_filtered_w_dist, median_d_loss_file)

        return med_filtered_w_dist, g_loss_list

    def generate_image(self, dir, iteration_num):
        bz = self.z_sampler(self.batch_size, self.z_dim)
        bx = self.sess.run(self.x_, feed_dict={self.z: bz})
        bx = self.x_sampler.data2img(bx)
        bx = grid_transform(bx, self.x_sampler.shape)
        imsave('{}/{}.png'.format(dir, iteration_num / 100), bx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='dcgan')
    parser.add_argument('--gpus', type=str, default='0')

    args = parser.parse_args()
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)
    xs = data.DataSampler()
    zs = data.NoiseSampler()
    d_net = model.Discriminator()
    g_net = model.Generator()
    wgan = WassersteinGAN(g_net, d_net, xs, zs, args.data, args.model, l_rate=5e-5, batch_size=64, iterations=400000)
    wgan.train()


