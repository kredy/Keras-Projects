'''

GAN using Keras functional API

'''

from keras.models import Model
from keras.datasets import mnist
from keras.layers import Dense, Activation, BatchNormalization, Flatten, Reshape, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


# Load the MNIST data
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Preprocess data
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = x_train.reshape(-1, 28, 28, 1)
    print(x_train.shape)
    return x_train


# Generator and Discriminator architecture
class GAN:

    def __init__(self, x_train):
        self.x_train = x_train
        self.img_shape = x_train.shape
        self.n_samples, self.img_h, self.img_w, self.img_c = self.img_shape

    def generator(self):
        noise_shape = (100,)
        inputs = Input(shape=noise_shape)
        layer = Dense(512)(inputs)
        layer = BatchNormalization()(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dense(512)(layer)
        layer = BatchNormalization()(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dense(1024)(layer)
        layer = BatchNormalization()(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dense(self.img_h*self.img_w)(layer)
        layer = Activation('tanh')(layer)
        layer = Reshape([self.img_w, self.img_h, self.img_c])(layer)
        model = Model(inputs=inputs, outputs=layer)
        model.summary()
        return model

    def discriminator(self):
        inputs = Input(shape=[self.img_w, self.img_h, self.img_c, ])
        layer = Flatten()(inputs)
        layer = Dense(1024)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dense(512)(layer)
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dense(1)(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs, outputs=layer)
        model.summary()
        return model


# Train the network
def train(generator, discriminator_train, gan_model, epochs, batch_size, x_train):

    half_batch = int(batch_size/2)

    for epoch in range(epochs):
        id_x = np.random.randint(0, 60000, half_batch)
        imgs = x_train[id_x]
        noise = np.random.normal(0, 1, (half_batch, 100))

        gen_imgs = generator.predict(noise)

        # Train discriminator
        loss_real = discriminator_train.train_on_batch(imgs, np.ones((half_batch, 1)))
        loss_fake = discriminator_train.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        loss = np.add(loss_real, loss_fake)*0.5

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        valid_y = np.array([1] * batch_size)
        gen_loss = gan_model.train_on_batch(noise, valid_y)
        
        print('{} Generator: loss={:0.3f} Discriminator: loss={:0.3f}, acc={:0.3f}'.format(epoch, gen_loss, loss[0], loss[1]))


# Show images
def show_imgs(generator):
    f, axes = plt.subplots(6, 6)
    noise = np.random.normal(0, 1, (36, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 127.5 * (gen_imgs + 127.5)
    count = 0
    for i in range(6):
        for j in range(6):
            axes[i, j].imshow(gen_imgs[count, :, :, 0])
            axes[i, j].axis('off')
            count = count + 1
    plt.show()


def main():
    optimizer = Adam(0.00025, 0.6)
    x_train = load_data()

    gan = GAN(x_train)

    generator = gan.generator()
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    k = Input(shape=[100, ])
    img = generator(k)

    discriminator = gan.discriminator()
    discriminator_train = discriminator
    discriminator_train.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    discriminator.trainable = False
    valid_img = discriminator(img)

    gan_model = Model(inputs=k, outputs=valid_img)
    gan_model.compile(loss='binary_crossentropy', optimizer=optimizer)

    train(generator, discriminator_train, gan_model, 10000, 128, x_train)
    
    show_imgs(generator)


if __name__ == '__main__':
    main()
