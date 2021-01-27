from tensorflow.keras import layers

import tensorflow as tf


class RoutingBlock(tf.keras.layers.Layer):
    def __init__(self, routes):
        super(RoutingBlock, self).__init__()
        self.routes = routes

    def call(self):
        raise NotImplementedError()

    def choose_route(self, inputs, routing_x):
        route_width = int(inputs.shape[-1] / self.routes)
        route = tf.argmax(routing_x, axis=-1)
        route = tf.one_hot(route, depth=self.routes)
        route_mask = tf.expand_dims(tf.expand_dims(tf.repeat(route, route_width, axis=1), axis=1), axis=1)
        route_mask = tf.repeat(route_mask, inputs.shape[1], axis=1)
        route_mask = tf.repeat(route_mask, inputs.shape[2], axis=2)
        x = tf.gather_nd(inputs, tf.where(route_mask == 1))
        x = tf.reshape(x, [-1, inputs.shape[1], inputs.shape[2], route_width])
        return x


class RandomRoutingBlock(RoutingBlock):
    def __init__(self, routes):
        super(RandomRoutingBlock, self).__init__(routes=routes)

    def call(self, inputs, is_training=True):
        routing_x = tf.random.uniform([tf.shape(inputs)[0], self.routes])

        x = self.choose_route(inputs, routing_x)
        return x, routing_x


class InformationGainRoutingBlock(RoutingBlock):
    def __init__(self, routes, dropout_rate):
        super(InformationGainRoutingBlock, self).__init__(routes=routes)

        self.dropout_rate = dropout_rate
        self.batch_norm = layers.BatchNormalization()
        self.conv = layers.Conv2D(64, (3, 3), (2, 2), padding='same')
        self.flatten = layers.GlobalAveragePooling2D()
        # self.dropout = layers.Dropout(self.dropout_rate)
        self.fc = layers.Dense(routes, activation=None)

    def call(self, inputs, training=None):
        routing_x = self.batch_norm(inputs)
        routing_x = self.conv(routing_x)
        routing_x = self.flatten(routing_x)
        # if training:
        #     routing_x = self.dropout(routing_x)
        routing_x = self.fc(routing_x)

        x = self.choose_route(inputs, routing_x)
        return x, routing_x
