from tensorflow.keras import layers

import tensorflow as tf


def routing_block(input_tensor,
                  num_routes,
                  stage
                  ):
    name_base = 'inf_gain_' + str(stage)
    x = layers.Conv2D(64, (3, 3),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name=f'{name_base}_conv')(input_tensor)
    x = layers.GlobalAveragePooling2D(name=f'{name_base}_gap')(x)
    x = layers.Dense(num_routes, activation=None,
                     name=f"{name_base}_fc")(x)
    return x


def routing_mask(information_gain_input_tensor, num_routes, feature_map_size):
    route_width = int(feature_map_size / num_routes)
    route = tf.argmax(information_gain_input_tensor, axis=-1)
    route_index_lower = tf.cast(tf.expand_dims(route * route_width, axis=-1), tf.int32)
    route_index_upper = tf.cast(tf.expand_dims((route + 1) * route_width, axis=-1), tf.int32)
    route_index = tf.expand_dims(tf.range(feature_map_size), axis=0)

    mask = tf.math.greater_equal(route_index, route_index_lower)
    mask = tf.logical_and(mask, tf.math.less(route_index, route_index_upper))
    mask = tf.expand_dims(tf.expand_dims(mask, 1), 1)

    return mask
