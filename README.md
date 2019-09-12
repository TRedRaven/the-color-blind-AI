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
`[[array([ 40, 239,  36]), 1],
  [array([223, 224, 212]), 4],
  [array([208,  27,  54]), 0],
  [array([237, 208, 233]), 4],
  [array([ 0, 11, 11]), 3],
  [array([  9, 210,  53]), 1],
  [array([ 5, 21, 53]), 3]]`<br/>
with the second entry in the second layer of the matrix being the correct
awnser.

## dependencies
- python 2.7
- numpy 1.16.5 or higher
