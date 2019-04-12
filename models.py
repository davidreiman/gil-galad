import tensorflow as tf
from tensorflow import keras as k
from tensorflow import layers as ly


from layers import *


class ResNet:
    def __init__(self, n_blocks, kernel_size, residual_filters, use_bias=True,
        activation='prelu', name='resnet'):

        self.kernel_size = kernel_size
        self.residual_filters = residual_filters
        self.use_bias = use_bias
        self.activation = activation
        self.n_blocks = n_blocks
        self.training = True
        self.name = name

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            x = conv_2d(
                x,
                kernel_size=self.kernel_size,
                filters=self.residual_filters,
                strides=1,
                activation='prelu',
            )

            x_ = tf.identity(x)

            for i in range(self.n_blocks):

                x = res_block_2d(
                    x,
                    kernel_size=self.kernel_size,
                    activation='prelu',
                    training=self.training
                )

            x = tf.add(x, x_)

            for j in range(2):
                x = subpixel_conv(
                    x,
                    upscale_ratio=2,
                    activation='prelu'
                )

            x = conv_2d(
                x,
                kernel_size=self.kernel_size,
                filters=3,
                strides=1,
                activation='sigmoid'
            )

            return x

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
