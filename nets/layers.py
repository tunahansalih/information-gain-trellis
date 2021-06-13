import tensorflow as tf
from tensorflow.keras import layers, regularizers


class ConvolutionalBlock(layers.Layer):
    def __init__(self, filters, kernel_size, padding="same"):
        super(ConvolutionalBlock, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, padding=padding)
        self.relu = layers.ReLU()
        self.pool = layers.MaxPool2D((2, 2))

    def call(self, inputs, is_training=True):
        x = self.conv(inputs)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ResNetBlock(layers.Layer):
    def __init__(self, stack, res_block, num_filters):
        self.stack = stack
        self.res_block = res_block

        if stack > 0 and res_block == 0:  # first layer but not first stack
            strides = 2  # downsample
        else:
            strides = 1

        self.resnet_layer_0 = ResNetLayer(filters=num_filters, strides=strides)
        self.resnet_layer_1 = ResNetLayer(filters=num_filters, activation=None)
        if stack > 0 and res_block == 0:  # first layer but not first stack
            # linear projection residual shortcut connection to match
            # changed dims
            self.resnet_layer_2 = ResNetLayer(
                filters=num_filters,
                kernel_size=1,
                strides=strides,
                activation=None,
                batch_normalization=False,
            )

    def call(self, inputs, is_training=True):
        x = inputs
        y = self.resnet_layer_0(x, is_training=is_training)
        y = self.resnet_layer_1(y, is_training=is_training)
        if self.stack > 0 and self.res_block == 0:
            x = self.resnet_layer_2(x)
        x = layers.add([x, y])
        x = layers.Activation("relu")(x)
        return x


class ResNetLayer(layers.Layer):
    def __init__(
        self,
        filters=16,
        kernel_size=3,
        strides=1,
        activation="relu",
        batch_normalization=True,
        conv_first=True,
    ):
        super().__init__()

        self.activation = activation
        self.batch_normalization = batch_normalization
        self.conv_first = conv_first
        self.conv = layers.Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(1e-4),
        )

    def call(self, inputs, is_training=True):
        x = inputs
        if self.conv_first:
            x = self.conv(x)
            if self.batch_normalization:
                x = layers.BatchNormalization()(x, is_training=is_training)
            if self.activation is not None:
                x = layers.Activation(self.activation)(x)
        else:
            if self.batch_normalization:
                x = layers.BatchNormalization()(x, is_training=is_training)
            if self.activation is not None:
                x = layers.Activation(self.activation)(x)
            x = self.conv(x)
        return x


class RandomRoutingBlock(layers.Layer):
    def __init__(self, routes):
        super(RandomRoutingBlock, self).__init__()
        self.routes = routes

    def call(self, inputs, is_training=True):
        routing_x = tf.random.uniform([tf.shape(inputs)[0], self.routes])
        return routing_x


class InformationGainRoutingBlock(layers.Layer):
    def __init__(self, routes):
        super(InformationGainRoutingBlock, self).__init__()
        self.routes = routes
        self.flatten = layers.Flatten()
        self.fc0 = layers.Dense(64, activation=tf.nn.relu)
        self.routing = layers.Dense(self.routes, activation=None)

    def call(self, inputs, is_training=True):
        x = self.flatten(inputs)
        x = self.fc0(x)
        x = self.routing(x)
        return x


class RoutingMaskLayer(layers.Layer):
    def __init__(self, routes, gumbel=False):
        super(RoutingMaskLayer, self).__init__()
        self.routes = routes
        self.gumbel = gumbel

    def __call__(self, inputs, routing_inputs, is_training=True):
        input_shape = tf.shape(inputs)
        route_width = int(inputs.shape[-1] / self.routes)

        if self.gumbel and is_training:
            routing_inputs = routing_inputs + self.sample_gumbel(
                tf.shape(routing_inputs)
            )

        route = tf.argmax(routing_inputs, axis=-1)
        route = tf.one_hot(route, depth=self.routes)

        route_mask = tf.repeat(route, repeats=route_width, axis=1)

        x = tf.transpose(inputs, [0, 3, 1, 2])
        x = tf.reshape(
            tf.boolean_mask(x, route_mask),
            [input_shape[0], route_width, input_shape[1], input_shape[2]],
        )
        x = tf.transpose(x, [0, 2, 3, 1])
        return x

    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        return -tf.math.log(
            -tf.math.log(tf.random.uniform(shape, minval=0, maxval=1) + eps) + eps
        )

    # def gumbel_softmax(self, logits, temperature, hard=False):
    #     gumbel_softmax_sample = logits + self.sample_gumbel(tf.shape(logits))
    #     y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    #     if hard:
    #         y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
    #         y = tf.stop_gradient(y_hard - y) + y
    #     return y
