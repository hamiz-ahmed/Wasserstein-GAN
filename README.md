# Wasserstein-GAN

This is the tensorflow implementation of WGAN algorithm. The purpose was to find out whether Wasserstein Distance could be used as a metric in determining quality of images on various configurations. To optimize hyper-parameters, Hyperband is used.

## Training
The training was primarily done on MNIST dataset. The output consisted of high quality images on low learning rates