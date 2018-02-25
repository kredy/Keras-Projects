'''

A simple Residual Neural Network architecture using Keras functional API

'''

import pickle
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D, Add
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ReduceLROnPlateau
import keras.backend as K
import tensorflow as tf



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



# ResNet Blocks
class Blocks():


	def __init__(self):
		pass


	# Identity shortcut block
	def IdentityBlock(self,input_matrix,filters,s1,s2,block_name):

		filters1, filters2 = filters

		nxt_layer = Conv2D(filters1, (s1,s2), padding='SAME',
			name=block_name+'_Conv2D_1')(input_matrix)
		nxt_layer = BatchNormalization(name=block_name+'_BatchNorm_1')(nxt_layer)
		nxt_layer = Activation('relu', name=block_name+'_ReLu_1')(nxt_layer)
		
		nxt_layer = Conv2D(filters2, (s1,s2), padding='SAME',
			name=block_name+'_Conv2D_2')(nxt_layer)
		nxt_layer = BatchNormalization(name=block_name+'_BatchNorm_2')(nxt_layer)
		nxt_layer = Activation('relu', name=block_name+'_ReLu_2')(nxt_layer)
		
		shortcut = input_matrix

		nxt_layer = Add()([nxt_layer,shortcut])
		nxt_layer = Activation('relu', name=block_name+'_ReLu_3')(nxt_layer)

		return nxt_layer


	# Convolution shortcut block
	def ConvBlock(self,input_matrix,filters,s1,s2,block_name):

		filters1, filters2 = filters

		nxt_layer = Conv2D(filters1, (s1,s2), padding='SAME',
			name=block_name+'_Conv2D_1')(input_matrix)
		nxt_layer = BatchNormalization(name=block_name+'_BatchNorm_1')(nxt_layer)
		nxt_layer = Activation('relu', name=block_name+'_ReLu_1')(nxt_layer)
		
		nxt_layer = Conv2D(filters2, (1,1), padding='SAME',
			name=block_name+'_Conv2D_2')(nxt_layer)
		nxt_layer = BatchNormalization(name=block_name+'_BatchNorm_2')(nxt_layer)
		nxt_layer = Activation('relu', name=block_name+'_ReLu_2')(nxt_layer)
		
		shortcut = Conv2D(filters2, (1,1), padding='SAME',
			name=block_name+'_Shortcut_Conv2D')(input_matrix)
		shortcut = BatchNormalization(name=block_name+'_Shortcut_BatchNorm')(shortcut)

		nxt_layer = Add()([shortcut,nxt_layer])
		nxt_layer = Activation('relu', name=block_name+'_ReLu_3')(nxt_layer)

		return nxt_layer


	# Dense Layer
	def DenseBlock(self,input_matrix,n_hidden,block_name,out=False):

		nxt_layer = Dense(n_hidden,name=block_name+'Dense')(input_matrix)
		nxt_layer = BatchNormalization(name=block_name+'_BatchNorm')(nxt_layer)
		
		if out:
			nxt_layer = Activation('softmax',name=block_name+'_Out_Layer')(nxt_layer)
		else:
			nxt_layer = Activation('relu',name=block_name+'_ReLu')(nxt_layer)
			nxt_layer = Dropout(0.5,name=block_name+'_Dropout')(nxt_layer)

		return nxt_layer



# Network architecture
def ResNET(blocks):

	with tf.name_scope('Inputs') as graph:
		inputs = Input(name='Inputs',shape=[32,32,3])

	with tf.name_scope('First_Conv_Block') as graph:
		layer = Conv2D(256,(2,2),padding='SAME',name='Conv2D_1')(inputs)

	with tf.name_scope('Second_Block') as graph:
		layer = blocks.IdentityBlock(layer,[512,256],2,2,'Block_2')
		layer = MaxPooling2D(pool_size=(2,2),name='MaxPooling2D_1')(layer)

	with tf.name_scope('Third_Block') as graph:
		layer = blocks.ConvBlock(layer,[256,128],2,2,'Block_3a')
		layer = blocks.IdentityBlock(layer,[128,128],2,2,'Block_3b')
		layer = blocks.IdentityBlock(layer,[128,128],2,2,'Block_3c')
		layer = MaxPooling2D(pool_size=(2,2),name='MaxPooling2D_2')(layer)

	with tf.name_scope('Fourth_Block') as graph:
		layer = blocks.ConvBlock(layer,[256,128],2,2,'Block_4a')
		layer = blocks.IdentityBlock(layer,[128,128],2,2,'Block_4b')
		layer = blocks.IdentityBlock(layer,[128,128],2,2,'Block_4c')
		layer = MaxPooling2D(pool_size=(2,2),name='MaxPooling2D_3')(layer)

	with tf.name_scope('Flatten') as graph:
		layer = Flatten()(layer)

	with tf.name_scope('FC1_Block') as graph:
		layer = blocks.DenseBlock(layer,2048,'FC_1')

	with tf.name_scope('FC2_Block') as graph:
		layer = blocks.DenseBlock(layer,2048,'FC_2')

	with tf.name_scope('Out_Block') as graph:
		layer = blocks.DenseBlock(layer,10,'Out',out=True)

	model = Model(inputs=inputs,outputs=layer)

	return model



def main():
	
	K.clear_session()
	batch_size = 128
	epochs = 50
	learning_rate = 0.0001
	Tboard = TensorBoard(log_dir="./ResNet_graph")
	reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.01,min_lr=0.00001,patience=5)

	data = LoadData('data.bin','labels.bin')
	x_train,x_dev,y_train,y_dev = data.train_dev_Set()
	x_test,y_test = data.test_Set('data_test.bin','labels_test.bin')
	y_train,y_dev,y_test = data.OneHot()

	blocks = Blocks()

	model = ResNET(blocks)
	model.compile(loss='categorical_crossentropy', 
		optimizer=Adam(lr=learning_rate),metrics=['accuracy'])
	model.summary()

	model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
		validation_data=(x_dev,y_dev),callbacks=[Tboard,reduce_lr])

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
	model.save_weights('small_ResNET_weights.hdf5')



if __name__ == '__main__':
	main()
