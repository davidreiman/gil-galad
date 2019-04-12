import tensorflow as tf
from tensorflow import keras as k
from tensorflow import layers as ly


def res_block_2d(x, kernel_size,  training, batch_norm=True):

    assert len(x.shape) == 4, "Input tensor must be 4-dimensional."

    filters = int(x.shape[3])

    y = ly.conv2d(
        inputs=x,
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same'
    )

    if batch_norm:
        y = ly.batch_normalization(y, training=training)

    y = k.layers.PReLU()(y)

    y = ly.conv2d(
        inputs=y,
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same'
    )

    if batch_norm:
        y = ly.batch_normalization(y, training=training)

    return tf.add(x, y)


def subpixel_conv(x, upscale_ratio, kernel_size=3):

    assert len(x.shape) == 4, "Input tensor must be 4-dimensional."
    assert isinstance(upscale_ratio, int), "Upscale ratio must be integer-valued."

    n_filters = int(x.shape[3])

    y = ly.conv2d(
        inputs=x,
        filters=n_filters*upscale_ratio**2,
        kernel_size=kernel_size,
        strides=1,
        padding='same'
    )

    y = tf.depth_to_space(y, block_size=upscale_ratio)

    return y


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

            x = ly.conv2d(
                inputs=x,
                filters=self.residual_filters,
                kernel_size=self.kernel_size,
                strides=1,
                padding='same'
            )
            x = k.layers.PReLU()(x)

            x_ = tf.identity(x)

            for i in range(self.n_blocks):

                x = res_block_2d(
                    x,
                    kernel_size=self.kernel_size,
                    training=self.training
                )

            x = tf.add(x, x_)

            for j in range(2):
                x = subpixel_conv(
                    x,
                    upscale_ratio=2,
                )
                x = k.layers.PReLU()(x)

            x = ly.conv2d(
                inputs=x,
                filters=3,
                kernel_size=self.kernel_size,
                strides=1,
                padding='same'
            )
            x = tf.sigmoid(x)

            return x

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]
