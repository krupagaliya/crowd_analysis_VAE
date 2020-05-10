import os
import logging
from keras.layers import Dense, Input, MaxPooling2D, UpSampling2D
from keras.layers import Conv2D, Flatten, Lambda, Dropout
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.losses import mse
from keras.utils import plot_model
from keras import backend as K
from datetime import datetime

logging.getLogger('tensorflow').disabled = True


class ModelController:
    def __init__(self, latent_dim, filters, kernel_size, above_dense):
        self.latent_dim = latent_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.above_dense = above_dense

    def sampling(self, args):
        z_mean, z_log_sigma = args
        batch = K.shape(z_mean)[0]
        epsilon = K.random_normal(shape=(batch, self.latent_dim),
                                  mean=0., stddev=1.)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def encoder_model(self, input_shape):
        # build encoder model

        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        for i in range(2):
            self.filters *= 2
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(x)
            # x = Dropout(.2)

        # shape info needed to build decoder model
        shape = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(self.above_dense, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        z = Lambda(self.sampling)([z_mean, z_log_var])

        # instantiate encoder model
        # encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder = Model(inputs, z, name='encoder')
        now = datetime.now()
        model_name  = str(now.day) + str(now.strftime("%X")) + 'enc'
        if not os.path.exists('model'):
            os.mkdir('model')

        plot_model(encoder, to_file='model/{}.png'.format(model_name), show_shapes=True)
        # print(encoder.summary())
        # print("---------------------------"*10)
        return inputs, shape, encoder, z_mean, z_log_var

    def decoder_model(self, shape):
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
        x = Reshape((shape[1], shape[2], shape[3]))(x)
        print("x shape", x.shape)

        for i in range(2):
            x = Conv2DTranspose(filters=self.filters,
                                kernel_size=self.kernel_size,
                                activation='relu',
                                strides=2,
                                padding='same')(x)
            self.filters //= 2

        outputs = Conv2DTranspose(filters=1,
                                  kernel_size=self.kernel_size,
                                  activation='sigmoid',
                                  padding='same',
                                  name='decoder_output')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        now = datetime.now()
        model_name = str(now.day) + str(now.strftime("%X")) + 'dec'
        # print(decoder.summary())
        # print("----------------------------"*10)
        plot_model(decoder, to_file='model/{}.png'.format(model_name), show_shapes=True)
        return decoder

    def deep_autoencoder(self, input_img):
        conv1 = Conv2D(filters=512, kernel_size=self.kernel_size, activation='relu', padding='same')(input_img)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters=768, kernel_size=self.kernel_size, activation='relu', padding='same')(pool1)
        conv3 = Conv2D(filters=1024, kernel_size=self.kernel_size, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # conv4 = Conv2D(filters=2048, kernel_size=self.kernel_size, activation='relu', padding='same')(pool2)

        #Deocder
        # conv5 = Conv2D(filters=2048, kernel_size=self.kernel_size, activation='relu', padding='same')(conv4)
        up1 = UpSampling2D((2, 2))(pool2)
        conv6 = Conv2D(filters=1024, kernel_size=self.kernel_size, activation='relu', padding='same')(up1)
        conv7 = Conv2D(filters=512, kernel_size=self.kernel_size, activation='relu', padding='same')(conv6)
        up2 = UpSampling2D((2, 2))(conv7)
        conv8 = Conv2D(filters=256, kernel_size=self.kernel_size, activation='relu', padding='same')(up2)
        conv9 = Conv2D(filters=64, kernel_size=self.kernel_size, activation='relu', padding='same')(conv8)
        decodeded = Conv2D(filters=1, kernel_size=self.kernel_size, activation='relu', padding='same')(conv9)
        return decodeded


if __name__ == '__main__':
    kernel_size = 3
    filters = 16
    latent_dim = 70
    epochs = 4
    above_dense = 64

    modelobj = ModelController(latent_dim, filters, kernel_size, above_dense)
    input_shape = (256, 344, 1)
    inputs, shape, encoder, z_mean, z_log_var = modelobj.encoder_model(input_shape)

    decoder = modelobj.decoder_model(shape)

    outputs = outputs = decoder(encoder(inputs))
    vae = Model(inputs, outputs, name='vae')

    vae.summary()

