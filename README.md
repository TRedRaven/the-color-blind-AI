# The color blind AI
This is a learning project where a ai must quess which of
five options i want it to give.

And thus this project will be heavely documented.

## function
The network will take 3 inputs with them representing R, G and B
respectively and it will have to out put if i gave it <br/>
red with `R>200, G<55, B<55` as inputs <br/>
green with `R<55, G>200, B<55` as inputs <br/>
blue with `R<55, G<55, B>200` as inputs <br/>
black with `R<55, G<55, B<55` as inputs <br/>
white with `R>200, G>200, B>200` as inputs <br/>

The network will have 1 hidden layer of 8 neurons and
will use the librarie numpy.

## datasets

To create the datasets and minibatches there will be a script to
make a dataset that looks like this: <br/>
`[[array([[227],
       [243],
       [242]]), 4], [array([[  6],
       [225],
       [ 20]]), 1], [array([[ 36],
       [  5],
       [241]]), 2], [array([[19],
       [10],
       [46]]), 3], [array([[  8],
       [ 34],
       [216]]), 2], [array([[253],
       [216],
       [220]]), 4], [array([[216],
       [248],
       [240]]), 4], [array([[248],
       [ 32],
       [ 44]]), 0], [array([[ 15],
       [212],
       [ 24]]), 1], [array([[239],
       [212],
       [216]]), 4]]`<br/>
with the second entry in the second layer of the matrix being the correct
awnser.

## dependencies
- python 2.7
- numpy 1.16.5 or higher

## discovery log
a log of all the stuff i found out while trying to get it right

commit 12: <br/>
in all previous attempts whenever i would call the feedforward method
i would get weird outputs such as a 2 dim array with in the 1st dim 8
arrays with 8 values which should have been a 1 dim array with 8 values
apparently this was caused by the biases being a 2 dim array with in the 1st dim arrays with 1 value. it would take that one value and multiply it with every value of the output of the np.dot() function
which would cause the weird output. so i changed:

- self.biases = [np.random.randn(y, 1) for y in self.layers[1:]] <br/>
to
- self.biases = [np.random.randn(1, y) for y in self.layers[1:]] <br/>

and now it gives good outputs
