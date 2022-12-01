import backprop_data
import numpy as np
import matplotlib.pyplot as plt
import backprop_network

########## Q2 ###########

#####################################  a #####################################
def a():
    training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

###################################### b ######################################
def b():

    def plot_accuracy(epochs_num, acc, learning_rates, title):

        for i in range(len(learning_rates)):
            plt.plot(np.arange(epochs_num), acc[i], label=str(learning_rates[i]))

        plt.xlabel('epochs')
        plt.ylabel(title)
        plt.legend()
        plt.show()

    training_data, test_data = backprop_data.load(10000,5000)
    learning_rates = [0.001, 0.01, 0.1, 1, 10, 100]
    training_accuracy = [None]*len(learning_rates)
    training_loss = [None]*len(learning_rates)
    test_accuracy = [None]*len(learning_rates)
    epochs_num = 30
    for i in range(len(learning_rates)):
        net = backprop_network.Network([784, 40, 10])
        training_accuracy[i],training_loss[i],test_accuracy[i] = net.SGD(training_data, epochs_num, 10, learning_rates[i], test_data)

    plot_accuracy(epochs_num, training_accuracy, learning_rates, 'training accuracy')
    plot_accuracy(epochs_num, training_loss, learning_rates, 'training loss')
    plot_accuracy(epochs_num, test_accuracy, learning_rates, 'test accuracy')

##################################### c #####################################
def c():
    training_data, test_data = backprop_data.load(50000,10000)
    net = backprop_network.Network([784, 40, 10])
    net.SGD(training_data, 30, 10, 0.1, test_data)

print("************************** a **************************")
a()
print("************************** b **************************")
b()
print("************************** c **************************")
c()