import matplotlib.pyplot as plt


def extract():

    training_data = open('logs/w_dist/reading.txt', 'r').read()
    #steps = open('logs/w_dist/reading.txt', 'r').read()
    #validation_data = open('validation_accuracy.txt', 'r').read()



    training_data = training_data.split('\n')

    del training_data[-1]

    training_data = [abs(float(var)) for var in training_data]
    steps = range(len(training_data))

#    del steps[-1]
 #   del training_data[-1]
  #  del validation_data[-1]

    plt.plot(steps, training_data, label='validation')
#    plt.plot(steps, training_data, label='training')

    plt.xlabel('epochs')
    plt.ylabel('w-dist')
    plt.legend(loc='lower right')
    plt.show()

extract()