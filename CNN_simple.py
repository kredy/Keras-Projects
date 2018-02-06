'''

Simple CNN to classify digits in the MNIST dataset 
using Keras functional API

'''


# Import the necessary modules
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Activation, Dense, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical


# Paramters
batch_size = 64
learning_rate = 0.001
epochs = 5


# Load the MNIST data
def load_data():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	y_train = to_categorical(y_train, num_classes=10)
	y_test = to_categorical(y_test, num_classes=10)
	x_train = x_train.reshape(-1,28,28,1)
	x_test = x_test.reshape(-1,28,28,1)
	x_train = x_train/255.0
	x_test = x_test/255.0
	return x_train, y_train, x_test, y_test


# ConvNET
def ConvNET():
	inputs = Input(name="Input",shape=[28,28,1])
	layer = Conv2D(64,(5,5),padding='same',name='Conv2D_1')(inputs)
	layer = Activation("relu",name="relu_activation_1")(layer)
	layer = MaxPooling2D(pool_size=(2,2),name='MaxPooling2D_1')(layer)
	layer = Conv2D(64,(5,5),padding='same',name='Conv2D_2')(layer)
	layer = Activation("relu",name="relu_activation_2")(layer)
	layer = MaxPooling2D(pool_size=(2,2),name='MaxPooling2D_2')(layer)
	layer = Flatten()(layer)
	layer = Dense(256,name='Dense_1')(layer)
	layer = Activation('relu',name='relu_activation_3')(layer)
	layer = Dense(256,name='Dense_2')(layer)
	layer = Activation('relu',name='relu_activation_4')(layer)
	out = Dense(10,name='Out_layer')(layer)
	out = Activation('softmax', name='softmax_activation')(out)
	model = Model(inputs=inputs,outputs=out)
	return model


# main function
def main():
	x_train,y_train,x_test,y_test = load_data()
	model = ConvNET()
	model.compile(loss="categorical_crossentropy", 
		optimizer=Adam(lr=learning_rate), metrics=["accuracy"])
	model.summary()
	model.fit(x=x_train, y=y_train, 
		batch_size=batch_size, epochs=epochs,
		callbacks=[keras.callbacks.TensorBoard(log_dir="./graph")])
	train_acc = model.evaluate(x=x_train,y=y_train,
		batch_size=100)
	test_acc = model.evaluate(x=x_test,y=y_test,
		batch_size=100)
	print("Training accuracy\n", train_acc)
	print('Testing accuracy\n', test_acc)

main()