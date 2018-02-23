# Keras-Projects
Neural networks in Keras using Tensorflow backend.
____

Experiments on various popular neural network architectures with fewer layers, fewer filters, and fewer hidden units.


* [CNN_simple.py](https://github.com/kredy/Keras-Projects/blob/master/CNN_simple.py)

  * Convolutional neural network using Keras functional API.
  * [Understanding Convolutional Neural Networks with A Mathematical Model](https://arxiv.org/abs/1609.04112v2)

* [LeNet-5ish.py](https://github.com/kredy/Keras-Projects/blob/master/LeNet-5ish.py)

  * LeNet-5 like neural network using Keras functional API.
  * [Gradient Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)

* [deeper_CNN.py](https://github.com/kredy/Keras-Projects/blob/master/deeper_CNN.py)

  * Deeper CNN (6 convolutional layers - 3 blocks) using Keras functional API.
  * VGG like architecture (much smaller) with Batch Normalization.
  * [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

* [small_ResNET.py](https://github.com/kredy/Keras-Projects/blob/master/small_ResNET.py)

  * A simple residual neural network with Identity blocks and Convolutional blocks.
  * [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
  * [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
  
* [small_DenseNet.py](https://github.com/kredy/Keras-Projects/blob/master/small_DenseNet.py)
  
  * A ConvNet similar to DenseNet. 
  * [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
  
*MNIST dataset taken from `keras.datasets`.*

*[The MNIST Database](http://yann.lecun.com/exdb/mnist/)*

*[CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)*
___

### Tested on:

- Python 3.5.2
- Tensorflow 1.5.0
- Keras 2.1.4
