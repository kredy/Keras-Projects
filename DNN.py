'''

Neural network using Keras functional API

'''

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, BatchNormalization, Dense, Activation, Dropout
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.utils import to_categorical


# Load the MNIST data
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    x_train = x_train/255.0
    x_test = x_test/255.0
    return x_train, y_train, x_test, y_test


# Network architecture
def network():
    inputs = Input(name='inputs', shape=[28*28, ])

    layer = Dense(1024, name='FC1')(inputs)
    layer = BatchNormalization(name='BC1')(layer)
    layer = Activation('relu', name='Act1')(layer)
    layer = Dropout(0.5, name='Dropout1')(layer)
    layer = Dense(768, name='FC2')(layer)
    layer = BatchNormalization(name='BC2')(layer)
    layer = Activation('relu', name='Act2')(layer)
    layer = Dropout(0.5, name='Dropout2')(layer)
    layer = Dense(512, name='FC3')(layer)
    layer = BatchNormalization(name='BC3')(layer)
    layer = Activation('relu', name='Act3')(layer)
    layer = Dropout(0.5, name='Dropout3')(layer)
    layer = Dense(768, name='FC4')(layer)
    layer = BatchNormalization(name='BC4')(layer)
    layer = Activation('relu', name='Act4')(layer)
    layer = Dropout(0.5, name='Dropout4')(layer)
    layer = Dense(1024, name='FC5')(layer)
    layer = BatchNormalization(name='BC5')(layer)
    layer = Activation('relu', name='Act5')(layer)
    layer = Dropout(0.5, name='Dropout5')(layer)
    layer = Dense(10, activation='softmax', name='Out')(layer)

    model = Model(inputs=inputs, outputs=layer)
    return model


def main():

    batch_size = 128
    epochs = 3
    TBoard = TensorBoard(log_dir='./DNN_graph')

    x_train, y_train, x_test, y_test = load_data()

    model = network()
    model.summary()

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, callbacks=[TBoard], validation_split=0.15)

    print(model.evaluate(x=x_test, y=y_test, batch_size=batch_size))

    model.save_weights('./DNN_weights.hdf5')


if __name__ == '__main__':
    main()
