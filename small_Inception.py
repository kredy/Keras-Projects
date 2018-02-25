'''

A small ConvNet similar to Inception (with dimensionality reduction) architecture

'''

import pickle
from keras.models import Model 
from keras.layers import Activation, LeakyReLU, Dense, Input, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Load the data from the pickled file
class LoadData():


	def __init__(self,train_data_name,train_labels_name,test_data_name,test_labels_name):
		
		# This method could be avoided and the data could be loaded using keras.datasets
		
		# Load the training data
		with open(train_data_name,'rb') as fo:
			X = pickle.load(fo)
		
		with open(train_labels_name,'rb') as fo:
			Y = pickle.load(fo)
		
		self.X = X
		self.Y = Y
		
		#Load the test data
		with open(test_data_name,'rb') as fo:
			X_test = pickle.load(fo)

		with open(test_labels_name,'rb') as fo:
			Y_test = pickle.load(fo)

		self.X_test = X_test
		self.Y_test = Y_test


	def train_dev_split(self):

		X_train,X_dev,Y_train,Y_dev = train_test_split(self.X,self.Y,
			test_size=0.15,random_state=3)
		
		self.Y_train = Y_train
		self.Y_dev = Y_dev

		return X_train, X_dev, self.X_test

	def oneHot(self):

		Y_train = to_categorical(self.Y_train,num_classes=10)
		Y_dev = to_categorical(self.Y_dev,num_classes=10)
		Y_test = to_categorical(self.Y_test,num_classes=10)

		return Y_train, Y_dev, Y_test



class Blocks():

	def __init__(self):
		pass


	def inceptionBlock(self,input_matrix,block_name):

		# Original inception used ReLU rather than LeakyReLU
		branch1 = Conv2D(16,(1,1),padding='SAME',
			name=block_name+'_A1_Conv')(input_matrix)
		branch1 = BatchNormalization(name=block_name+'_A1_BatchNorm')(branch1)
		branch1 = LeakyReLU(alpha=0.0001,name=block_name+'_A1_Activation')(branch1)
		
		branch2 = Conv2D(16,(1,1),padding='SAME',
			name=block_name+'_B1_Conv')(input_matrix)
		branch2 = Conv2D(32,(3,3),padding='SAME',
			name=block_name+'_B2_Conv')(branch2)
		branch2 = BatchNormalization(name=block_name+'_B3_BatchNorm')(branch2)
		branch2 = LeakyReLU(alpha=0.0001,name=block_name+'_B4_Activation')(branch2)

		branch3 = Conv2D(4,(1,1),padding='SAME',
			name=block_name+'_C1_Conv')(input_matrix)
		branch3 = Conv2D(8,(5,5),padding='SAME',
			name=block_name+'_C2_Conv')(branch3)
		branch3 = BatchNormalization(name=block_name+'_C3_BatchNorm')(branch3)
		branch3 = LeakyReLU(alpha=0.0001,name=block_name+'_C4_Activation')(branch3)

		branch4 = MaxPooling2D(pool_size=(2,2),padding='SAME',strides=(1,1),
			name=block_name+'_D1_MaxPool')(input_matrix)
		branch4 = Conv2D(8,(1,1),padding='SAME',
			name=block_name+'_D2_Conv')(branch4)
		branch4 = BatchNormalization(name=block_name+'_D3_BatchNorm')(branch4)
		branch4 = LeakyReLU(name=block_name+'_D4_Activation')(branch4)

		nxt_layer = Concatenate()([branch1,branch2,branch3,branch4])

		return nxt_layer


	def denseBlock(self,input_matrix,n_hidden,block_name,out=False):

		nxt_layer = Dense(n_hidden,name=block_name+'_Dense')(input_matrix)
		nxt_layer = BatchNormalization(name=block_name+'_Batch_Norm')(nxt_layer)

		if out:
			nxt_layer = Activation('softmax',name=block_name+'_Activation')(nxt_layer)
		else:
			nxt_layer = LeakyReLU(alpha=0.0001,name=block_name+'_Activation')(nxt_layer)
			nxt_layer = Dropout(0.7,name=block_name+'_Dropout')(nxt_layer)

		return nxt_layer



def Inception(blocks):

	inputs = Input(name='Inputs',shape=[32,32,3])

	layer = Conv2D(32,(3,3),padding='SAME',name='Conv2D')(inputs)
	layer = BatchNormalization(name='BatchNorm')(layer)
	layer = LeakyReLU(name='Activation')(layer)
	layer = MaxPooling2D(pool_size=(2,2),name='MaxPool_1')(layer)

	layer = blocks.inceptionBlock(layer,'Inception_1')

	layer = blocks.inceptionBlock(layer,'Inception_2')

	layer = MaxPooling2D(pool_size=(2,2),name='MaxPool_2')(layer)

	layer = blocks.inceptionBlock(layer,'Inception_3')

	layer = GlobalAveragePooling2D(name='GlobalAveragePooling')(layer)

	layer = blocks.denseBlock(layer,1024,'FC')

	layer = blocks.denseBlock(layer,10,'Out',out=True)

	model = Model(inputs=inputs,outputs=layer)

	return model



def main():

	learning_rate = 0.0001
	batch_size = 128
	epochs = 50
	Tboard = TensorBoard(log_dir="./Inception_graph")
	reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.01,min_lr=0.000001,patience=3)
	E_stop = EarlyStopping(monitor='val_loss',min_delta=0.0001)

	data = LoadData('data.bin','labels.bin','data_test.bin','labels_test.bin')
	X_train, X_dev, X_test = data.train_dev_split()
	Y_train, Y_dev, Y_test = data.oneHot()

	blocks = Blocks()
	model = Inception(blocks)
	model.summary()

	model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=learning_rate),
		metrics=['accuracy'])
	model.fit(x=X_train,y=Y_train,batch_size=batch_size,epochs=epochs,
		validation_data=(X_dev,Y_dev),callbacks=[Tboard,reduce_lr,E_stop])

	xlst = [X_train,X_dev,X_test]
	ylst = [Y_train,Y_dev,Y_test]

	for i, (x,y) in enumerate(zip(xlst,ylst)):

		accr = model.evaluate(x=x,y=y,batch_size=batch_size)

		if i == 0:
			print('-'*50)
			print('\nTraining set\n  Loss: {:03f}\n  Accuracy: {:0.3f}\n'.format(accr[0],accr[1]))
			print('-'*50)

		elif i == 1:
			print('-'*50)
			print('\nDev set\n  Loss: {:03f}\n  Accuracy: {:0.3f}\n'.format(accr[0],accr[1]))
			print('-'*50)

		else:
			print('-'*50)
			print('\nTest set\n  Loss: {:03f}\n  Accracy: {:0.3f}\n'.format(accr[0],accr[1]))
			print('-'*50)


	model.save_weights('small_Inception_weights.hdf5')

if __name__ == '__main__':
	main()