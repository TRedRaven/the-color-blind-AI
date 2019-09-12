#!/usr/bin/python

""" the main file for the colorblind AI, to run it simply use `python main.py`.
    the program also takes arguments:
        set `-T <int>` to set the amount of times you want to train the network,
        use `-D` for debug mode, this will print thing like the current weights and biases.
        set `--data <RRGB>, <G>, <B>` with R, G and B being numbers between 0 and 255 to
        give the network a custom set inputs to see what it thinks is the right awnser,
"""

# imports
import numpy as np # numpy is for the matrix multiplecation and array generatation
import optparse # used for making the network a bit more controleble
import gen_minibatch as gmb # the minibatch creation script

# miscellaneous fucntions
def sigmoid(x):
    """will take a int and normalize it returning something inbetween 0 and 1"""
    return 1 / (1 + np.exp(-x))

class neural_net():
    """neural_net takes 1 arugment and that is the options from optparse"""
    def __init__(self, options):
        """creating weights and biases and running"""
        # if it should display debug text
        self.debug = options.debug
        # the amount it will train
        self.train_amount = options.train_amount
        # the mini batch data, default size is 100
        self.minibatch = gmb.update_minibatch(1)

        # initializing layers as 3 inputs 1 hidden with 8 neurons and 5 outputs
        self.layers = (3, 8, 5)

        # generating the weights and biases
        self.biases = [np.random.randn(1, y) for y in self.layers[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(self.layers[:-1], self.layers[1:])]

        # debug printing the biases and weights
        if self.debug:
            print "starting biases: %s\n" % self.biases
            print "starting weights: %s\n" % self.weights

        if not self.train_amount == None:
            for i in range(self.train_amount):
                self.train()

    def feedforward(self, activation):
        """this will do in this case two matrix multiplecations
        bases on the column of activations given in the arugment `activation`
        and self.weights as a matrix to compair it to and add the biases.
        """
        for b, w in zip(self.biases, self.weights):
            activation = sigmoid(np.dot(activation, w) + b)
        return activation

    def train(self):
        """this will train the network for the amount of time set in the `-T` argument"""
        for data in self.minibatch:
            input = sigmoid(data[0]) # putting the normalized rgb value in input
            awnser = data[1] # the index of the value that should be 1
            output = self.feedforward(input)
            if self.debug:
                print "input | awnser: %s | %s" % (input, awnser)
                print "output: %s" % output

# only run when this file is being called specificly. so it doesn't
# trigger when the file is being imported
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-D', action="store_true", dest="debug", default=False,
                      help="print variables such as weights and biases")
    parser.add_option('-T', action="store", dest="train_amount", default=None, type="int",
                      help="if you want to train the ai (wich you will have to do in the bigining)")
    parser.add_option('--data', action="store", dest="data", default='255, 255, 255',
                      type="str", help="the data you want to ai to seekout. syntax to R,G,B")

    options, remainder = parser.parse_args()
    net = neural_net(options)
