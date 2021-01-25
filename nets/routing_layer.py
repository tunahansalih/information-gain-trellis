from tensorflow.keras import layers

import tensorflow as tf


class RoutingLayer(tf.keras.layers.Layer):
    def __init__(self, routes):
        super(RoutingLayer, self).__init__()
        self.routes = routes
        self.conv = layers.Conv2D(64, (3, 3),
                                  strides=(2, 2),
                                  padding='valid',
                                  kernel_initializer='he_normal',
                                  )
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(routes, activation=None)

    def call(self, inputs):
        routing_x = self.conv(inputs)
        routing_x = self.avg_pool(routing_x)
        routing_x = self.fc(routing_x)

        route_width = int(inputs.shape[-1] / self.routes)
        route = tf.argmax(routing_x, axis=-1)
        route = tf.one_hot(route, depth=self.routes)
        route_mask = tf.expand_dims(tf.expand_dims(tf.repeat(route, route_width, axis=1), axis=1), axis=1)
        route_mask = tf.repeat(route_mask, inputs.shape[1], axis=1)
        route_mask = tf.repeat(route_mask, inputs.shape[2], axis=2)

        x = tf.gather_nd(inputs, tf.where(route_mask == 1))
        x = tf.reshape(x, [-1, inputs.shape[1], inputs.shape[2], route_width])
        return x, routing_x
