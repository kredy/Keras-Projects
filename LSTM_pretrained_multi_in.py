'''

LSTM multi-in single out network using pre-trained word vectors.

'''


import numpy as np
import pandas as pd
from keras.layers import Activation, Dense, LSTM, Embedding, Dropout, BatchNormalization, Input, Add
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split


max_num_words = 30000
max_seq_len = 100
batch_size = 128
epochs = 30
TBoard = TensorBoard(log_dir='./LSTM_pretrained_multi_in')


class LoadData:

    def __init__(self, filename):
        # -----------
        # Data from https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products
        # -----------
        self.filename = filename
        df = pd.read_csv(self.filename, delimiter=',')
        df = df[['reviews.text', 'reviews.title' ,'reviews.rating']]
        df = df.dropna()
        x = df.drop('reviews.rating', axis=1)
        y = df['reviews.rating']
        y = y.astype('int64') - 1
        self.x = x
        self.y = y

    # Split into training and test set
    def split(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.15, random_state=10)
        y_train = to_categorical(y_train, num_classes=5)
        y_test = to_categorical(y_test, num_classes=5)
        # Two different inputs
        x_train_1 = x_train['reviews.text'].values
        x_train_2 = x_train['reviews.title'].values
        x_test_1 = x_test['reviews.text'].values
        x_test_2 = x_test['reviews.title'].values
        x_train_1 = x_train_1.reshape(-1,1)
        x_train_2 = x_train_2.reshape(-1,1)
        return x_train_1, x_train_2, x_test_1, x_test_2, y_train, y_test


class TextPreprocessing:

    def __init__(self):
        pass

    # Tokenize the input data and pad the sequences
    def tokens(self, x_train_1, x_train_2, x_test_1, x_test_2):
        tokenizer_1 = Tokenizer(num_words=max_num_words)
        tokenizer_2 = Tokenizer(num_words=max_num_words)
        x_train_1 = x_train_1.tolist()
        x_train_2 = x_train_2.tolist()
        tokenizer_1.fit_on_texts(texts=x_train_1)
        tokenizer_2.fit_on_texts(texts=x_train_2)
        x_train_1 = tokenizer_1.texts_to_sequences(texts=x_train_1)
        x_train_2 = tokenizer_2.texts_to_sequences(texts=x_train_2)
        word_index_1 = tokenizer_1.word_index
        self.word_index_1 = word_index_1
        word_index_2 = tokenizer_2.word_index
        self.word_index_2 = word_index_2
        x_test_1 = x_test_1.tolist()
        x_test_2 = x_test_2.tolist()
        x_test_1 = tokenizer_1.texts_to_sequences(texts=x_test_1)
        x_test_2 = tokenizer_2.texts_to_sequences(texts=x_test_2)
        x_train_1 = pad_sequences(x_train_1, maxlen=max_seq_len)
        x_train_2 = pad_sequences(x_train_2, maxlen=max_seq_len)
        x_test_1 = pad_sequences(x_test_1, maxlen=max_seq_len)
        x_test_2 = pad_sequences(x_test_2, maxlen=max_seq_len)
        return x_train_1, x_train_2, x_test_1, x_test_2

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

    # Embeddings matrices
    def emb_matrix(self):
        num_words_1 = min(max_num_words, len(self.word_index_1) + 1)
        embedding_matrix_1 = np.zeros((num_words_1, 50))
        for word, i in self.word_index_1.items():
            if i >= max_num_words:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix_1[i] = embedding_vector
        num_words_2 = min(max_num_words, len(self.word_index_2) + 1)
        embedding_matrix_2 = np.zeros((num_words_2, 50))
        for word, i in self.word_index_2.items():
            if i >= max_num_words:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix_2[i] = embedding_vector
        return embedding_matrix_1, embedding_matrix_2, num_words_1, num_words_2


# Network architecture
def RNN(embeddings_matrix_1, embeddings_matrix_2, num_words_1, num_words_2):
    
    inputs_1 = Input(shape=[max_seq_len, ], name='inputs_1')
    e_1 = Embedding(num_words_1, 50, weights=[embeddings_matrix_1], 
        input_length=max_seq_len, trainable=False)(inputs_1)
    layer_1 = LSTM(32, dropout=0.5)(e_1)
    layer_1 = Dense(128, name='FC1')(layer_1)
    layer_1 = BatchNormalization(name='BN1')(layer_1)
    layer_1 = Activation('relu', name='ReLU1')(layer_1)
    layer_1 = Dropout(0.2, name='Dropout1')(layer_1)
    
    inputs_2 = Input(shape=[max_seq_len, ], name='inputs_2')
    e_2 = Embedding(num_words_2, 50, weights=[embeddings_matrix_2], 
        input_length=max_seq_len, trainable=False)(inputs_2)
    layer_2 = LSTM(32, dropout=0.5)(e_2)
    layer_2 = Dense(128, name='FC2')(layer_2)
    layer_2 = BatchNormalization(name='BN2')(layer_2)
    layer_2 = Activation('relu', name='ReLU2')(layer_2)
    layer_2 = Dropout(0.4, name='Dropout2')(layer_2)
    
    layer = Add()([layer_1, layer_2])
    
    layer = Dense(256, name='FC3')(layer)
    layer = BatchNormalization(name='BN3')(layer)
    layer = Activation('relu', name='ReLU3')(layer)
    layer = Dropout(0.4, name='Dropout3')(layer)
    layer = Dense(256, name='FC4')(layer)
    layer = BatchNormalization(name='BN4')(layer)
    layer = Activation('relu', name='ReLU4')(layer)
    layer = Dropout(0.4, name='Dropout4')(layer)
    layer = Dense(5, name='outlayer')(layer)
    layer = Activation('softmax', name='out_Act')(layer)
    
    model = Model(inputs=[inputs_1, inputs_2], outputs=layer)
    
    return model


def main():
    data = LoadData('7817_1.csv')
    x_train_1, x_train_2, x_test_1, x_test_2, y_train, y_test = data.split()

    text_preprocessing = TextPreprocessing()
    x_train_1, x_train_2, x_test_1, x_test_2 = text_preprocessing.tokens(
        x_train_1, x_train_2, x_test_1, x_test_2)
    text_preprocessing.emb_index('./word_vectors/glove.6B.50d.txt')
    embedding_matrix_1, embedding_matrix_2, num_words_1, num_words_2 = text_preprocessing.emb_matrix()

    model = RNN(embedding_matrix_1, embedding_matrix_2, num_words_1, num_words_2)
    model.summary()

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=[x_train_1, x_train_2], y=y_train, validation_split=0.15, batch_size=batch_size, 
        epochs=epochs, callbacks=[TBoard])
    
    print(model.evaluate(x=[x_test_1, x_test_2], y=y_test, batch_size=batch_size))
    
    model.save_weights('LSTM_pretrained_multi_in_weights.hdf5')


if __name__ == '__main__':
    main()
