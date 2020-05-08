import os
import logging
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.losses import mse
from keras.utils import plot_model
from keras import backend as K
from datetime import datetime
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("-ld", "--latent_dim", type=int)
# parser.add_argument("-f", "--filters", type=int)
# parser.add_argument("-k", "--kernel", type=int)
# parser.add_argument("-ad", "--above_dense", type=int)
# args = parser.parse_args()

# latent_dim = args.latent_dim
# filters = args.filters
# kernel_size = args.kernel
# above_dense = args.above_dense

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
        # input_shape = x_train.shape[1:]
        inputs = Input(shape=input_shape, name='encoder_input')
        x = inputs
        for i in range(2):
            self.filters *= 2
            x = Conv2D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       activation='relu',
                       strides=2,
                       padding='same')(x)

        # shape info needed to build decoder model
        shape = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(self.above_dense, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        z = Lambda(self.sampling)([z_mean, z_log_var])

        # instantiate encoder model
        # encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        encoder = Model(inputs, z_mean, name='encoder')
        now = datetime.now()
        model_name  = str(now.day) + str(now.strftime("%X")) + 'enc'
        if not os.path.exists('model'):
            os.mkdir('model')

        plot_model(encoder, to_file='model/{}.png'.format(model_name), show_shapes=True)
        return inputs, shape, encoder

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
        plot_model(decoder, to_file='model/{}.png'.format(model_name), show_shapes=True)
        return decoder


# if __name__ == '__main__':
#
#     inputs, shape, encoder = encoder_model(input_shape)
#     decoder = decoder_model(shape)
#     outputs = decoder(encoder(inputs))
#
#     vae = Model(inputs, outputs, name='vae')
#     print(vae.summary())