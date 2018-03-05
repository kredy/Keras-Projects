'''

Bidirectional LSTM using Keras functional API

'''

import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Activation, BatchNormalization, Embedding, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


# Load the data
def load_data():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=2500)
    return (x_train, y_train), (x_test, y_test)


# RNN structure
def bi_rnn(maxlen):
    inputs = Input(shape=[maxlen], name='inputs')
    layer = Embedding(2500, 50, input_length=maxlen)(inputs)
    layer = Bidirectional(LSTM(64))(layer)
    layer = Dense(256, name='FC1')(layer)
    layer = BatchNormalization(name='BatchNorm1')(layer)
    layer = Activation('relu', name='Activation1')(layer)
    layer = Dropout(0.5, name='Dropout1')(layer)
    layer = Dense(256, name='FC2')(layer)
    layer = BatchNormalization(name='BatchNorm2')(layer)
    layer = Activation('relu', name='Activation2')(layer)
    layer = Dropout(0.5, name='Dropout2')(layer)
    layer = Dense(1, name='Out')(layer)
    layer = Activation('sigmoid', name='sigmoid')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


def main():
    batch_size = 128
    epochs = 30
    learning_rate = 0.0001
    max_len = 150
    TBoard = TensorBoard(log_dir='./biRNN')

    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = pad_sequences(x_train, maxlen=max_len)
    x_test = pad_sequences(x_test, maxlen=max_len)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    model = bi_rnn(max_len)
    model.summary()
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, 
        validation_split=0.15, callbacks=[TBoard])
    
    xlst = [x_train, x_test]
    ylst = [y_train, y_test]
    for i, (x,y) in enumerate(zip(xlst, ylst)):
        accr = model.evaluate(x, y, batch_size=batch_size)
        if i == 0:
            print('-'*50)
            print('\nTraining set\n  Loss: {:03f}\n  Accuracy: {:0.3f}\n'.format(accr[0],accr[1]))
            print('-'*50)
        else:
            print('-'*50)
            print('\nTest set\n  Loss: {:03f}\n  Accuracy: {:0.3f}\n'.format(accr[0],accr[1]))
            print('-'*50)

    model.save_weights('bidirectional_LSTM_weights.hdf5')

if __name__ == '__main__':
    main()
