#!/usr/bin/python

""" the main file for the colorblind AI, to run it simply use `python main.py`.
    the program also takes arguments:
        set `-T <int>` to set the amount of times you want to train the network,
        use `-D` for debug mode, this will print thing like the current weights and biases.
        set `--data <RRGB>, <G>, <B>` with R, G and B being numbers between 0 and 255 to
        give the network a custom set inputs to see what it thinks is the right awnser,

	Added to test branch.

"""

# imports
import numpy as np # numpy is for the matrix multiplecation and array generatation
import optparse # used for making the network a bit more controleble
import gen_minibatch as gmb # the minibatch creation script

# miscellaneous fucntions
def sigmoid(x):
    """will take a int and normalize it returning something inbetween 0 and 1"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(z):
    """will return the derivative of the sigmoid value"""
    return sigmoid(z)*(1-sigmoid(z))

def cost_derivative(a, y):
    """this will return the derivative of the cost"""
    return (a-y)

class neural_net():
    """neural_net takes 1 arugment and that is the options from optparse"""
    def __init__(self, options):
        """creating weights and biases and running"""
        # if it should display debug text
        self.debug = options.debug
        # the amount it will train
        self.train_amount = options.train_amount
        # the mini batch data, default size is 100
        # print self.minibatch
        # initializing layers as 3 inputs 1 hidden with 8 neurons and 5 outputs
        self.layers = (3, 8, 5)

        # generating the weights and biases
        self.biases = [np.random.randn(1, y) for y in self.layers[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(self.layers[:-1], self.layers[1:])]

        # debug printing the biases and weights
        # if self.debug:
            # print "starting biases: %s\n" % self.biases
            # print "starting weights: %s\n" % self.weights
        self.minibatch = [[[1, 0, 0], [1, 0, 0, 0, 0]],
                          [[0, 1, 0], [0, 1, 0, 0, 0]],
                          [[0, 0, 1], [0, 0, 1, 0, 0]],
                          [[1, 1, 1], [0, 0, 0, 1, 0]],
                          [[0, 0, 0], [0, 0, 0, 0, 1]]]

        if not self.train_amount == None:
            for i in range(self.train_amount):
                self.train()

        if self.debug:
            # print "input | awnser: %s | %s" % (input, awnser)
            output = self.feedforward([1,0,0])
            print output
            print "awnser: %s" % [1,0,0,0,0]
            while True:
                thing = [0] * 3
                print "the array: %s" % thing
                thing[0] = int(raw_input("array index 0 = "))
                print "the array: %s" % thing
                thing[1] = int(raw_input("array index 1 = "))
                print "the array: %s" % thing
                thing[2] = int(raw_input("array index 2 = "))
                print "the array: %s" % thing
                output = self.feedforward(thing)
                print output

    def backprop(self, x, y):
        """this method will go back and see how much every weight and biases
        changes the output and it that change is a reinforcing any corrent activation
        and supressing any incorrect activation.
        if takes two arguments:
        x:
        """
        # chreating empty copys of the biases and weights
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # print "starting nabla_b: %s" % nabla_b
        # print "starting nabla_w: %s" % nabla_w
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        # print "starting activations: %s" % activations
        zs = [] # list to store all the z vectors, layer by layer
        # print "_______________entering first for loop_______________"
        for b, w in zip(self.biases, self.weights):
            # print "w: %s" % w
            # print "b: %s" % b
            z = np.dot(activation, w) + b
            zs.append(z)
            # print "zs: %s" % zs
            activation = sigmoid(z)
            # print "activation: %s" % activation
            activations.append(activation)
        # print "_______________exit first for loop_______________"
        # print "activations after forward feeding: %s" % activations
        # print "output: %s" % activations[-1]
        # print "awnser: %s" % y
        # backward pass
        delta = cost_derivative(activations[-1], y) * \
            sigmoid_derivative(zs[-1])
        # print "delta: %s" % delta

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot( activations[-2].transpose(), delta)
        # print "nable_b: %s" % nabla_b

        for l in xrange(2, len(self.layers)):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(delta, self.weights[-l+1].transpose()) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l].transpose())
        # print "nabla_b after backprop: %s" % nabla_b
        # print "nabla_w after backprop: %s" % nabla_w
        return (nabla_b, nabla_w)

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
        # for data in self.minibatch:
        #     input = data[0] # putting the normalized rgb value in input
        #     awnser = data[1] # the index of the value that should be 1
        #     output = self.feedforward(input)
        #     print "cost = %s - %s" % (awnser, output)
        #     cost = awnser - output
        #     print "cost: %s" % cost
        #     nabla_b, nabla_w = self.backprop(input, awnser)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # print "_______________entering train method_______________"

        for x, y in self.minibatch:
            # print "_______________exiting train method_______________"
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # print "_______________entering train method_______________"
            # print "return from backprop(): %s | %s" % (delta_nabla_b, delta_nabla_w)
            # print "nabla_w: %s " % nabla_w
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            # print "nabla_w: %s " % nabla_w
            # print "q|a: %s|%s" % (x, y)
        # print "applying weights: %s" % nabla_w
        # print "before: %s" % self.weights
        # print "self.weights before gradiant decent aplied: %s" % self.weights
        self.weights = [w-nw for w, nw in zip(self.weights, nabla_w)]
        # print "after: %s" % self.weights
        self.biases = [b-nb for b, nb in zip(self.biases, nabla_b)]
        # print "self.weights after gradiant decent aplied: %s" % self.weights
        # print "_______________exiting train method_______________"


# only run when this file is being called specificly. so it doesn't
# trigger when the file is being imported
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-D', '--debug', action="store_true", dest="debug", default=False,
                      help="print variables such as weights and biases")
    parser.add_option('-T', '--train', action="store", dest="train_amount", default=None, type="int",
                      help="if you want to train the ai (wich you will have to do in the bigining)")
    parser.add_option('-r', '--run', action="store", dest="run_data", default='255, 255, 255',
                      type="str", help="the data you want to ai to determan. syntax to R,G,B")

    options, remainder = parser.parse_args()
    net = neural_net(options)
