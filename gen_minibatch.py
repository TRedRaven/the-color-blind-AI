import random
import numpy as np

def sigmoid(x):
    """will take a int and normalize it returning something inbetween 0 and 1"""
    return 1 / (1 + np.exp(-x))

def update_minibatch(size=100):
    """will generate a new mini batch of colors and the anwsers,
    there is one argument,
    size: the size of the column of colors and awnsers. it defaults to 100
    """

    # making a column of rgb values
    batch = []
    for i in range(size):
        awnser = random.randint(0,4)
        # if the awnser is white
        if awnser == 4:
            RGB = np.ones(3)

        # if the awnser is black
        elif awnser == 3:
            RGB = np.zeros(3)
        elif awnser in range(3):
            RGB = np.zeros(3)
            RGB[awnser] = 1
        awnserlist = [0] * 5
        awnserlist[awnser] = 1
        batch.append([RGB, awnserlist])
        print "this is the batch: %s" % batch
    return batch


if __name__ == "__main__":
    print update_minibatch(10)
