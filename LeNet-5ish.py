'''

LeNet-5 like neural network using Keras functional API

'''

# Import necessary libraries
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Network paramaters
train_dev_split = 0.15
learning_rate = 0.0001
batch_size=128
epochs = 50


# Load the data
class LoadData():

	def __init__(self):
		pass

	def load_from_dataset(self):
		(x,y), (x_test,y_test) = mnist.load_data()
		x_test = x_test/225.0
		x_test = x_test.reshape(-1,28,28,1)
		return x,y,x_test,y_test

	def train_dev_split(self,train_dev_split,x,y):
		x_train,x_dev,y_train,y_dev = train_test_split(x,y,
			test_size=train_dev_split,random_state=20)
		x_train = x_train/225.0
		x_train = x_train.reshape(-1,28,28,1)
		x_dev = x_dev/225.0
		x_dev = x_dev.reshape(-1,28,28,1)
		return x_train,y_train,x_dev,y_dev

	def Onehot(self,y_train,y_dev,y_test):
		y_train = to_categorical(y_train,num_classes=10)
		y_dev = to_categorical(y_dev,num_classes=10)
		y_test = to_categorical(y_test,num_classes=10)
		return y_train,y_dev,y_test


# ConvNet model
def LeNet():

	inputs = Input(name='inputs',shape=[28,28,1])
	
	layer = Conv2D(6,(5,5),padding='same',name='Conv2D_1')(inputs)
	layer = Activation('relu',name='Activation_1')(layer)
	layer = MaxPooling2D(pool_size=(2,2),name='MaxPooling2D_1')(layer)
	layer = Conv2D(16,(5,5),padding='valid',name='Conv2D_2')(layer)
	layer = Activation('relu',name='Activation_2')(layer)
	layer = MaxPooling2D(pool_size=(2,2),name='MaxPooling2D_2')(layer)
	layer = Flatten()(layer)
	layer = Dense(120,name='Dense_1')(layer)
	layer = Activation('relu',name='Activation_3')(layer)
	layer = Dense(84,name='Dense_2')(layer)
	layer = Activation('relu',name='Activation_4')(layer)
	layer = Dense(10,name='out_layer')(layer)
	layer = Activation('softmax',name='Activation_sotmax')(layer)
	
	model = Model(inputs=inputs,outputs=layer)

	return model 


# Main function
def main():

	loaddata = LoadData()
	x,y,x_test,y_test = loaddata.load_from_dataset()
	x_train,y_train,x_dev,y_dev = loaddata.train_dev_split(train_dev_split,x,y)
	y_train,y_dev,y_test = loaddata.Onehot(y_train,y_dev,y_test)

	model = LeNet()
	model.compile(loss='categorical_crossentropy',
			optimizer=Adam(lr=learning_rate),metrics=['accuracy'])
	model.fit(x=x_train,y=y_train,
			batch_size=batch_size,epochs=epochs,
			callbacks=[TensorBoard(log_dir="./LeNet-5ish")])
	
	train_acc = model.evaluate(x=x_train,y=y_train,batch_size=batch_size)
	print('Training set accuracy: ',train_acc)
	
	dev_acc = model.evaluate(x=x_dev,y=y_dev,batch_size=batch_size)
	print('Validation set accuracy: ',dev_acc)

	test_acc = model.evaluate(x=x_test,y=y_test,batch_size=batch_size)
	print('Testing set accuracy: ',test_acc)
	
	# Save trained weights
	model.save_weights('LeNet-5ish_weights.hdf5')
	

main()

