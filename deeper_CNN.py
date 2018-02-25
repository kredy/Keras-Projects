'''

Deeper CNN using Keras Functional API

'''

import pickle
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, BatchNormalization, Dropout 
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical


# Load the data and convert it into the required format
class LoadData():


	def __init__(self,data_filename,labels_filename):
		
		# Load CIFAR-10 from a pickeld file
		# The same could also be done using keras.datasets
		self.data_filename = data_filename
		self.labels_filename = labels_filename
		with open(data_filename,'rb') as fo:
			x = pickle.load(fo)
		with open(labels_filename,'rb') as fo:
			y = pickle.load(fo)
		x = x/255.0
		self.x = x
		self.y = y


	def train_dev_Set(self):

		# Spilliting into train and dev set
		x_train,x_dev,y_train,y_dev = train_test_split(self.x,self.y,
			test_size=0.15,random_state=25)
		self.y_train = y_train
		self.y_dev = y_dev
		
		return x_train, x_dev, y_train, y_dev


	def test_Set(self,test_data_filename,test_labels_filename):
		
		# Load test set from pickled file
		with open(test_data_filename, 'rb') as fo:
			x_test = pickle.load(fo)
		with open(test_labels_filename, 'rb') as fo:
			y_test = pickle.load(fo)
		x_test = x_test/255.0
		self.y_test = y_test
		
		return x_test, y_test


	def OneHot(self):

		# Concert to One-Hot encoding
		yoh_train = to_categorical(self.y_train, num_classes=10)
		yoh_dev = to_categorical(self.y_dev, num_classes=10)
		yoh_test = to_categorical(self.y_test, num_classes=10)
		
		return yoh_train, yoh_dev, yoh_test



# Convolution block
def ConvBlock(input_matrix,filters,s1,s2,pool_size,block_name):

	nxt_layer = Conv2D(filters, (s1,s2), padding='SAME',
		name=block_name+'_Conv2D_1')(input_matrix)
	nxt_layer = BatchNormalization(name=block_name+'_BatchNorm_1')(nxt_layer)
	nxt_layer = Activation('relu', name=block_name+'_ReLu_1')(nxt_layer)
	
	nxt_layer = Conv2D(filters, (s1,s2), padding='SAME',
		name=block_name+'_Conv2D_2')(nxt_layer)
	nxt_layer = BatchNormalization(name=block_name+'_BatchNorm_2')(nxt_layer)
	nxt_layer = Activation('relu', name=block_name+'_ReLu_2')(nxt_layer)
	
	nxt_layer = MaxPooling2D(pool_size=(pool_size,pool_size),
		name=block_name+'_MaxPool')(nxt_layer)

	return nxt_layer



# Dense Layer
def DenseBlock(input_matrix,n_hidden,block_name,out=False):

	nxt_layer = Dense(n_hidden,name=block_name+'Dense')(input_matrix)
	nxt_layer = BatchNormalization(name=block_name+'_BatchNorm')(nxt_layer)
	
	if out:
		nxt_layer = Activation('softmax',name=block_name+'_Out_Layer')(nxt_layer)
	else:
		nxt_layer = Activation('relu',name=block_name+'_ReLu')(nxt_layer)
		nxt_layer = Dropout(0.5,name=block_name+'_Dropout')(nxt_layer)

	return nxt_layer



# ConvNET model
def ConvNET():
	
	# Input Layer
	inputs = Input(name='inputs',shape=[32,32,3])

	# Block 1
	layer = ConvBlock(inputs,128,3,3,2,'Block_1')
	
	# Block 2
	layer = ConvBlock(layer,128,3,3,2,'Block_2')
	
	# Block 3
	layer = ConvBlock(layer,64,3,3,2,'Block_3')
	
	# Flatten
	layer = Flatten()(layer)

	# FC 1
	layer = DenseBlock(layer,1024,'FC_1')

	# FC 2
	layer = DenseBlock(layer,1024,'FC_2')

	# Out Layer
	layer = DenseBlock(layer,10,'Out_Block',out=True)

	model = Model(inputs=inputs,outputs=layer)

	return model



def main():
	
	batch_size = 128
	epochs = 100
	learning_rate = 0.0001
	Tboard = keras.callbacks.TensorBoard(log_dir="./graph")
	reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.01,min_lr=0.00001)

	data = LoadData('data.bin','labels.bin')
	x_train,x_dev,y_train,y_dev = data.train_dev_Set()
	x_test,y_test = data.test_Set('data_test.bin','labels_test.bin')
	y_train,y_dev,y_test = data.OneHot()

	model = ConvNET()
	model.compile(loss='categorical_crossentropy', 
		optimizer=Adam(lr=learning_rate),metrics=['accuracy'])
	model.summary()

	model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
		validation_data=(x_dev,y_dev),callbacks=[Tboard,reduce_lr])

	print('-'*50)
	
	x_lst = [x_train,x_dev,x_test]
	y_lst = [y_train,y_dev,y_test]

	for i, (x, y) in enumerate(zip(x_lst,y_lst)):
		accr = model.evaluate(x=x, y=y, batch_size=128)
		if i == 0:
			print('Training set\n  Loss: {:f}, Accuracy: {:0.3f}'.format(accr[0],accr[1]))
			print('-'*50)
		elif i == 1:
			print('Dev set\n  Loss: {:f}, Accuracy: {:0.3f}'.format(accr[0],accr[1]))
			print('-'*50)
		else:
			print('Test set\n  Loss: {:f}, Accuracy: {:0.3f}'.format(accr[0],accr[1]))
			print('-'*50)

	# Save trained weights
	model.save_weights('Deeper_CNN_weights.hdf5')



if __name__ == '__main__':
	main()
