import tensorflow as tf
from tensorflow.keras import layers, regularizers


class ConvolutionalBlock(layers.Layer):
    def __init__(self, filters, kernel_size, padding="same"):
        super(ConvolutionalBlock, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, padding=padding)
        self.relu = layers.ReLU()
        self.pool = layers.MaxPool2D((2, 2))

    def call(self, inputs, training=True):
        x = self.conv(inputs)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ResNetBlock(layers.Layer):
    def __init__(self, stack, res_block, num_filters):
        super().__init__()
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
        self.addition = layers.Add()
        self.relu = layers.Activation("relu")

    def call(self, inputs, training=True):
        x = inputs
        y = self.resnet_layer_0(x, training=training)
        y = self.resnet_layer_1(y, training=training)
        if self.stack > 0 and self.res_block == 0:
            x = self.resnet_layer_2(x)
        x = self.addition([x, y])
        x = self.relu(x)
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
        self.batch_norm_layer = layers.BatchNormalization()
        self.activation = layers.Activation(self.activation)

    def call(self, inputs, training=True):
        x = inputs
        if self.conv_first:
            x = self.conv(x)
            if self.batch_normalization:
                x = self.batch_norm_layer(x, training=training)
            if self.activation is not None:
                x = self.activation(x)
        else:
            if self.batch_normalization:
                x = self.batch_norm_layer(x, training=training)
            if self.activation is not None:
                x = self.activation(x)
            x = self.conv(x)
        return x


class RandomRoutingBlock(layers.Layer):
    def __init__(self, num_routes):
        super(RandomRoutingBlock, self).__init__()
        self.num_routes = num_routes

    def call(self, inputs, training=True):
        routing_x = tf.random.uniform([tf.shape(inputs)[0], self.num_routes])
        return routing_x


class InformationGainRoutingBlock(layers.Layer):
    def __init__(self, num_routes):
        super(InformationGainRoutingBlock, self).__init__()
        self.num_routes = num_routes
        self.flatten = layers.Flatten()
        self.fc0 = layers.Dense(64, activation=tf.nn.relu)
        self.routing = layers.Dense(self.num_routes, activation=None)

    def call(self, inputs, training=True):
        x = self.flatten(inputs)
        x = self.fc0(x)
        x = self.routing(x)
        return x


class RoutingMaskLayer(layers.Layer):
    def __init__(self, routes, gumbel=False):
        super(RoutingMaskLayer, self).__init__()
        self.routes = routes
        self.gumbel = gumbel

    def __call__(self, inputs, routing_inputs, training=True):
        input_shape = tf.shape(inputs)
        route_width = int(inputs.shape[-1] / self.routes)

        if self.gumbel and training:
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
