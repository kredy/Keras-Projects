'''

A small ConvNET much similar to DenseNet using Keras functional API

'''

import pickle
from keras.layers import Dense, Input, Activation, BatchNormalization, Dropout, LeakyReLU 
from keras.layers import Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


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



class Blocks():

	def __init__(self):
		pass


	# Conv block
	def ConvBlock(self,input_matrix,filters,s1,s2,block_name):

		nxt_layer = Conv2D(filters, (s1,s2), padding='SAME',
			name=block_name+'_Conv2D')(input_matrix)
		nxt_layer = BatchNormalization(name=block_name+'_BatchNorm')(nxt_layer)
		nxt_layer = LeakyReLU(alpha=0.0001,name=block_name+'_LeakyReLU')(nxt_layer)

		return nxt_layer


	# Dense block
	def DenseBlock(self,input_matrix,n_hidden,block_name,out=False):

			nxt_layer = Dense(n_hidden,name=block_name+'_Dense')(input_matrix)
			nxt_layer = BatchNormalization(name=block_name+'_BatchNorm')(nxt_layer)
			
			if out:
				nxt_layer = Activation('softmax',name=block_name+'_Out_Layer')(nxt_layer)
			else:
				nxt_layer = LeakyReLU(alpha=0.0001,name=block_name+'_Activation')(nxt_layer)
				nxt_layer = Dropout(0.5,name=block_name+'_Dropout')(nxt_layer)

			return nxt_layer


# Network architecture
def DenseNet(blocks):

	inputs = Input(name='Inputs',shape=[32,32,3])

	con_layer = Conv2D(32,(3,3),padding='SAME',name='Conv2D_1')(inputs)
	con_layer = BatchNormalization(name='BatchNorm_1')(con_layer)
	con_layer = LeakyReLU(alpha=0.0001,name='LeakyReLU_1')(con_layer)

	layer = blocks.ConvBlock(con_layer,32,2,2,'Block_A')
	layer = Concatenate()([layer,con_layer])

	layer = MaxPooling2D(pool_size=(2,2),name='MaxPool_1')(layer)

	con_layer = blocks.ConvBlock(layer,32,2,2,'Block_B1')
	layer = blocks.ConvBlock(con_layer,32,2,2,'Block_B2')
	layer = Concatenate()([layer,con_layer])

	layer = MaxPooling2D(pool_size=(2,2),name='MaxPool_2')(layer)
	
	con_layer = blocks.ConvBlock(layer,32,2,2,'Block_C1')
	layer = blocks.ConvBlock(con_layer,32,2,2,'Block_C2')
	layer = Concatenate()([layer,con_layer])

	layer = MaxPooling2D(pool_size=(2,2),name='MaxPool_3')(layer)

	layer = GlobalAveragePooling2D(name='Global_Avg_Pool')(layer)

	layer = blocks.DenseBlock(layer,1024,'Block_D')

	layer = blocks.DenseBlock(layer,1024,'Block_E')

	layer = blocks.DenseBlock(layer,10,'Block_F',out=True)

	model = Model(inputs=inputs, outputs=layer)

	return model



# Main
def main():
	
	batch_size = 128
	epochs = 50
	learning_rate = 0.0001
	Tboard = TensorBoard(log_dir="./DenseNet_graph")
	reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.01,min_lr=0.00001,patience=3)

	data = LoadData('data.bin','labels.bin')
	x_train, x_dev, y_train, y_dev = data.train_dev_Set()
	x_test, y_test = data.test_Set('data_test.bin','labels_test.bin')
	y_train, y_dev, y_test = data.OneHot()

	blocks = Blocks()

	model = DenseNet(blocks)

	model.summary()

	model.compile(loss='categorical_crossentropy', 
		optimizer=Adam(lr=learning_rate),metrics=['accuracy'])

	model.fit(x=x_train,y=y_train,batch_size=batch_size,epochs=epochs,
		validation_data=(x_dev,y_dev), callbacks=[Tboard,reduce_lr])

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
	model.save_weights('small_DenseNET_weights.hdf5')



if __name__ == '__main__':
	main()
