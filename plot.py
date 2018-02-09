"""
This file is used for plotting the saved results as readings.txt in wgans
"""

import matplotlib.pyplot as plt
from scipy import signal


def extract():

    data_good_config = open('HyperbandData/Trained_Models/7.353859474380557e-05_184/w_dist/reading.txt', 'r').read()
    data_orig_config = open('HyperbandData/Trained_Models/5e-05_64/w_dist/reading.txt', 'r').read()
    data_bad_config = open('HyperbandData/Trained_Models/0.010833716738437808_169/w_dist/reading.txt', 'r').read()
    data_good_config2 = open('HyperbandData/Trained_Models/0.0001494673269900912_77/w_dist/reading.txt', 'r').read()
    # data_good_config = open('/scratch/mahmed/DL_LAB project/Original-Config/Wasserstein-GAN/logs/mnist/0.010833716738437808_169/w_dist/reading.txt', 'r').read()
    # data_bad_config =  open('/scratch/mahmed/DL_LAB project/Original-Config/Data/Original_config_data/logs/w_dist/reading.txt', 'r').read()



    #steps = open('logs/w_dist/reading.txt', 'r').read()
    #validation_data = open('validation_accuracy.txt', 'r').read()



    # training_data_orig = training_data_orig.split('\n')
    # training_data_good = training_data_good.split('\n')
    #
    # del training_data_orig[-1]
    # del training_data_good[-1]
    #
    # training_data_orig = [float(var) for var in training_data_orig]
    # training_data_good = [float(var) for var in training_data_good]
    # steps = range(len(training_data_orig))
    # steps_2 = range(len(training_data_good))
    #
    # med_filtered_loss_orig = signal.medfilt(training_data_orig)
    # med_filt_good = signal.medfilt(training_data_good)

#    del steps[-1]
 #   del training_data_orig[-1]
  #  del validation_data[-1]

    steps1, data_orig_config = process_data_list(data_orig_config)
    steps2, data_good_config = process_data_list(data_good_config)
    steps3, data_good_config2 = process_data_list(data_good_config2)
    steps4, data_bad_config = process_data_list(data_bad_config)

    # plt.plot(steps1, data_orig_config, label='L-rate:5e-5, B-size:64')
    plt.plot(steps1, data_orig_config, label='Original Config')
    plt.plot(steps2, data_good_config, label='Very good Config')
    plt.plot(steps3, data_good_config2, label='Good Config')
    plt.plot(steps4, data_bad_config, label='Bad Config')

    plt.xlabel('epochs')
    plt.ylabel('wasserstein distance')
    plt.legend(loc='lower right')
    plt.show()

def process_data_list(data_list):
    data_list = data_list.split('\n')
    del data_list[-1]
    data_list = [float(var) for var in data_list]
    steps = len(data_list)
    med_filter = signal.medfilt(data_list)

    return range(steps), med_filter

extract()