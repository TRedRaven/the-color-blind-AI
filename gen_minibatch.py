import random
import numpy as np

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
            RGB = np.random.randint(low=200, high=255, size=3)

        # if the awnser is black
        elif awnser == 3:
            RGB = np.random.randint(low=0, high=55, size=3)
        elif awnser in range(3):
            RGB = np.random.randint(low=0, high=55, size=3)
            RGB[awnser] = random.randint(200, 255)
        batch.append([RGB, awnser])
    return batch


if __name__ == "__main__":
    print update_minibatch(10)
