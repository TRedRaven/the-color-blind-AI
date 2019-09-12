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

class neural_net():
    """neural_net takes 1 arugment and that is the options from optparse"""
    def __init__(self, options):
        """creating weights and biases and running"""
        # if it should display debug text
        self.debug = options.debug
        # the amount it will train
        self.train_amount = options.train_amount
        # the mini batch data, default size is 100
        self.minibatch = gmb.update_minibatch()

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
    print "minibatch: %s" % net.minibatch
