'''

LSTM using pre-trained word vectors.

'''


import numpy as np
from keras.layers import Activation, Dense, LSTM, Embedding, Dropout, BatchNormalization, Input
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split


max_num_words = 30000
max_seq_len = 100
batch_size = 256
epochs = 30
TBoard = TensorBoard(log_dir='./LSTM_pretrained')


class LoadData:

    def __init__(self, filename):
        self.filename = filename
        with open(filename, 'r') as fo:
            lines = fo.readlines()
            text = []
            labels = []
            for i in lines:
                text.append(i[:-5])
                labels.append(int(i[-2]))
            self.x = text
            self.y = labels

    # Split into training and test set
    def split(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.15, random_state=1)
        return x_train, x_test, y_train, y_test


class TextPreprocessing:

    def __init__(self):
        pass

    # Tokenize the input data and pad the sequences
    def tokens(self, x_train, x_test):
        tokenizer = Tokenizer(num_words=max_num_words)
        tokenizer.fit_on_texts(texts=x_train)
        x_train = tokenizer.texts_to_sequences(texts=x_train)
        word_index = tokenizer.word_index
        self.word_index = word_index
        x_test = tokenizer.texts_to_sequences(texts=x_test)
        x_train = pad_sequences(x_train, maxlen=max_seq_len)
        x_test = pad_sequences(x_test, maxlen=max_seq_len)
        return x_train, x_test

    # Getting the embeddings from the file
    def emb_index(self, filename):
        embeddings_index = {}
        with open(filename, 'r') as fo:
            for line in fo:
                values = line.split()
                word = values[0]
                coeffs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coeffs
        self.embeddings_index = embeddings_index

    # Embeddings matrix
    def emb_matrix(self):
        num_words = min(max_num_words, len(self.word_index) + 1)
        embedding_matrix = np.zeros((num_words, 50))
        for word, i in self.word_index.items():
            if i >= max_num_words:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix, num_words


# Network architecture
def RNN(embeddings_matrix, num_words):
    inputs = Input(shape=[max_seq_len, ], name='inputs')
    e = Embedding(num_words, 50, weights=[embeddings_matrix], 
        input_length=max_seq_len, trainable=False)(inputs)
    layer = LSTM(128, return_sequences=True, recurrent_dropout=0.4)(e)
    layer = LSTM(128, dropout=0.5)(layer)
    layer = Dense(64, name='FC1')(layer)
    layer = BatchNormalization(name='BN1')(layer)
    layer = Activation('relu', name='ReLU1')(layer)
    layer = Dropout(0.2, name='Dropout1')(layer)
    layer = Dense(128, name='FC2')(layer)
    layer = BatchNormalization(name='BN2')(layer)
    layer = Activation('relu', name='ReLU2')(layer)
    layer = Dropout(0.4, name='Dropout2')(layer)
    layer = Dense(1, name='outlayer')(layer)
    layer = Activation('sigmoid', name='out_Act')(layer)
    model = Model(inputs=inputs, outputs=layer)
    return model


def main():
    data = LoadData('./data.txt')
    x_train, x_test, y_train, y_test = data.split()

    text_preprocessing = TextPreprocessing()
    x_train, x_test = text_preprocessing.tokens(x_train, x_test)
    text_preprocessing.emb_index('./word_vectors/glove.6B.50d.txt')
    embeddings_matrix, num_words = text_preprocessing.emb_matrix()

    model = RNN(embeddings_matrix, num_words)
    model.summary()

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, validation_split=0.15, batch_size=batch_size,
              epochs=epochs, callbacks=[TBoard])

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

if __name__ == '__main__':
    main()
