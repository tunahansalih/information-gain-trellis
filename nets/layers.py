from tensorflow.keras import layers

import tensorflow as tf


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
    def __init__(self, routes):
        super(RoutingMaskLayer, self).__init__()
        self.routes = routes

    def __call__(self, inputs, routing_inputs):
        input_shape = tf.shape(inputs)
        route_width = int(inputs.shape[-1] / self.routes)

        route = tf.argmax(routing_inputs, axis=-1)
        route = tf.one_hot(route, depth=self.routes)

        route_mask = tf.repeat(route, repeats=route_width, axis=1)

        x = tf.transpose(inputs, [0, 3, 1, 2])
        x = tf.reshape(tf.boolean_mask(x, route_mask), [input_shape[0], route_width, input_shape[1], input_shape[2]])
        x = tf.transpose(x, [0, 2, 3, 1])
        return x


def sample_gumbel(shape, eps=1e-20):
    U = tf.random.uniform(shape, minval=0, maxval=1)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
