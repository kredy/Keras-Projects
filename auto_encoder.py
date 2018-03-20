'''

Convolutional autoencoder on MNIST dataset using Keras functional API

'''


from keras.datasets import mnist
from keras.models import Model
from keras.layers import Activation, Input, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# Parameters
batch_size = 128
epochs = 3
Tboard = TensorBoard(log_dir='./autoencoder_graph')

# Load the MNIST data
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_train = x_train/255.0
    x_test = x_test/255.0
    return x_train, y_train, x_test, y_test


# Autoencoder
def auto_encoder():
    # Encoder
    inputs = Input(name='inputs', shape=[28,28,1,])
    layer = Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), padding='valid', name='Conv2D_1')(inputs)
    layer = BatchNormalization(name='BN_1')(layer)
    layer = Activation('relu', name='relu_1')(layer)
    layer = Conv2D(filters=6, kernel_size=(5,5), strides=(1,1), padding='valid', name='Conv2D_2')(layer)
    layer = BatchNormalization(name='BN_2')(layer)
    layer = Activation('relu', name='relu_2')(layer)
    layer = Conv2D(filters=6, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='Conv2D_3')(layer)
    layer = BatchNormalization(name='BN_3')(layer)
    layer = Activation('relu', name='relu_3')(layer)
    encoder = Model(inputs=inputs, outputs=layer)
    # Decoder
    l_inputs = Input(name='l_inputs', shape=[18,18,6,])
    layer = Conv2DTranspose(filters=6, kernel_size=(3,3), strides=(1,1), padding='valid', name='deconv2d_1')(l_inputs)
    layer = BatchNormalization(name='BN_4')(layer)
    layer = Activation('relu', name='relu_4')(layer)
    layer = Conv2DTranspose(filters=6, kernel_size=(5, 5), strides=(1, 1), padding='valid', name='deconv2d_2')(layer)
    layer = BatchNormalization(name='BN_5')(layer)
    layer = Activation('relu', name='relu_5')(layer)
    layer = Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(1, 1), padding='valid', name='deconv2d_3')(layer)
    layer = Activation('relu', name='relu_6')(layer)
    decoder = Model(inputs=l_inputs, outputs=layer)
    # Encoder + Decoder
    model = Model(inputs=inputs, outputs=decoder(encoder(inputs)))
    return encoder, decoder, model


def main():
    x_train, y_train, x_test, y_test = load_data()
    
    encoder, decoder, model = auto_encoder()
    encoder.summary()
    decoder.summary()
    model.summary()
    
    model.compile(optimizer=Adam(), loss='mse')
    model.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, callbacks=[Tboard])
    
    gen_imgs = model.predict(x_test, batch_size=batch_size)
    
    # Visualisation of the generation images and comparision with the test images
    rn_num = np.random.randint(10000)
    gen_imgs = gen_imgs*255.0
    gen_img = gen_imgs[rn_num]
    x_test = x_test*255.0
    test_img = x_test[rn_num]
    test_img = test_img.reshape(28,28)
    gen_img = gen_img.reshape(28,28)
    # Show generated image
    plt.imshow(gen_img)
    plt.show()
    # Show test image
    plt.imshow(test_img)
    plt.show()
    
    # Save weights of encoder, decoder and the whole model
    encoder.save_weights('encoder_weights.hdf5')
    decoder.save_weights('decoder_weights.hdf5')
    model.save_weights('autoencoder_weights.hdf5')

    # Save architecture
    encoder_yaml = encoder.to_yaml()
    with open('encoder_string.yaml', 'w') as fo:
        fo.write(encoder_yaml)
    decoder_yaml = decoder.to_yaml()
    with open('decoder_string.yaml', 'w') as fo:
        fo.write(decoder_yaml)
    model_yaml = model.to_yaml()
    with open('model_string.yaml', 'w') as fo:
        fo.write(model_yaml)


if	__name__ == '__main__':
	main()
