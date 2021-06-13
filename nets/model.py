from enum import Enum
import tensorflow as tf

from nets.layers import (
    InformationGainRoutingBlock,
    RandomRoutingBlock,
    ConvolutionalBlock,
    ResNetBlock,
    ResNetLayer,
    RoutingMaskLayer,
)
from tensorflow.keras.layers import BatchNormalization


class Routing(Enum):
    NO_ROUTING = 0
    RANDOM_ROUTING = 1
    INFORMATION_GAIN_ROUTING = 2


class InformationGainRoutingModel(tf.keras.models.Model):
    def __init__(self, config):
        super(InformationGainRoutingModel, self).__init__()

        self.conv_block_0 = ConvolutionalBlock(
            filters=config["CNN_0"], kernel_size=(5, 5), padding="same"
        )
        if config["USE_ROUTING"]:
            self.routing_block_0 = InformationGainRoutingBlock(
                routes=config["NUM_ROUTES_0"]
            )
            self.random_routing_block_0 = RandomRoutingBlock(
                routes=config["NUM_ROUTES_0"]
            )
            self.routing_mask_layer_0 = RoutingMaskLayer(
                routes=config["NUM_ROUTES_0"], gumbel=config["ADD_GUMBEL_NOISE"]
            )
        self.batch_norm_0 = BatchNormalization()

        self.conv_block_1 = ConvolutionalBlock(
            filters=config["CNN_1"], kernel_size=(5, 5), padding="same"
        )
        if config["USE_ROUTING"]:
            self.routing_block_1 = InformationGainRoutingBlock(
                routes=config["NUM_ROUTES_1"]
            )
            self.random_routing_block_1 = RandomRoutingBlock(
                routes=config["NUM_ROUTES_1"]
            )
            self.routing_mask_layer_1 = RoutingMaskLayer(
                routes=config["NUM_ROUTES_1"], gumbel=config["ADD_GUMBEL_NOISE"]
            )
        self.batch_norm_1 = BatchNormalization()

        self.conv_block_2 = ConvolutionalBlock(
            filters=config["CNN_2"], kernel_size=(5, 5), padding="same"
        )
        self.batch_norm_2 = BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()

        self.fc_0 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)
        self.do_0 = tf.keras.layers.Dropout(config["DROPOUT_RATE"])
        self.fc_1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.do_1 = tf.keras.layers.Dropout(config["DROPOUT_RATE"])
        self.fc_2 = tf.keras.layers.Dense(config["NUM_CLASSES"])

    def call(self, inputs, routing: Routing, temperature=1, is_training=True):
        x = self.conv_block_0(inputs)
        x = self.batch_norm_0(x, training=is_training)

        if routing == Routing.RANDOM_ROUTING:
            routing_0 = self.random_routing_block_0(x)
        elif routing == Routing.INFORMATION_GAIN_ROUTING:
            routing_0 = self.routing_block_0(x, is_training=is_training) / temperature
        elif routing == Routing.NO_ROUTING:
            routing_0 = None
        else:
            routing_0 = None

        x = self.conv_block_1(x)
        if routing_0 is not None:
            x = self.routing_mask_layer_0(x, routing_0, is_training=is_training)
        x = self.batch_norm_1(x, training=is_training)

        if routing == Routing.RANDOM_ROUTING:
            routing_1 = self.random_routing_block_1(x)
        elif routing == Routing.INFORMATION_GAIN_ROUTING:
            routing_1 = self.routing_block_1(x, is_training=is_training) / temperature
        elif routing == Routing.NO_ROUTING:
            routing_1 = None
        else:
            routing_1 = None

        x = self.conv_block_2(x)
        if routing_1 is not None:
            x = self.routing_mask_layer_1(x, routing_1, is_training=is_training)
        x = self.batch_norm_2(x, training=is_training)

        x = self.flatten(x)

        x = self.fc_0(x)
        if is_training:
            x = self.do_0(x)
        x = self.fc_1(x)
        if is_training:
            x = self.do_1(x)
        x = self.fc_2(x)

        return routing_0, routing_1, x


class InformationGainRoutingLeNetModel(tf.keras.models.Model):
    def __init__(self, config):
        super(InformationGainRoutingLeNetModel, self).__init__()

        self.config = config

        self.conv_block_0 = ConvolutionalBlock(
            filters=config["CNN_0"],
            kernel_size=(5, 5),
            l2_weight_decay=config["L2_WEIGHT_DECAY"],
            padding="same",
        )

        self.routing_block_0 = InformationGainRoutingBlock(
            routes=config["NUM_ROUTES_0"], l2_weight_decay=config["L2_WEIGHT_DECAY"]
        )
        self.conv_blocks_1 = []
        for i in range(config["NUM_ROUTES_0"]):
            self.conv_blocks_1.append(
                ConvolutionalBlock(
                    filters=config["CNN_1"] // config["NUM_ROUTES_0"],
                    kernel_size=(5, 5),
                    l2_weight_decay=config["L2_WEIGHT_DECAY"],
                    padding="same",
                )
            )

        self.flatten = tf.keras.layers.Flatten()

        self.fc_0 = tf.keras.layers.Dense(
            100,
            activation=tf.nn.relu,
            kernel_regularizer=tf.keras.regularizers.l2(config["L2_WEIGHT_DECAY"]),
        )
        self.fc_1 = tf.keras.layers.Dense(config["NUM_CLASSES"])

    def call(self, inputs, temperature=1, is_training=True):
        x = self.conv_block_0(inputs)

        routing_0 = self.routing_block_0(x, is_training=is_training) / temperature

        routes_0 = []
        x_routed = 0
        for route in range(self.config["NUM_ROUTES_0"]):
            selection = (tf.argmax(routing_0, axis=-1) == route)[
                :, tf.newaxis, tf.newaxis, tf.newaxis
            ]
            x_routed += tf.where(
                condition=selection,
                x=self.conv_blocks_1[route](x),
                y=tf.zeros_like(selection),
            )

        x = self.flatten(x_routed)

        x = self.fc_0(x)
        x = self.fc_1(x)

        return routing_0, x


class InformationGainRoutingResNetModel(tf.keras.models.Model):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if (self.config.RESNET_DEPTH - 2) % 6 != 0:
            raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((self.config.RESNET_DEPTH - 2) / 6)

        self.resnet_layer_1 = ResNetLayer()

        self.resnet_blocks = []
        for stack in range(3):
            for res_block in range(num_res_blocks):
                resnet_block = ResNetBlock(stack, res_block, num_filters)
                self.resnet_blocks.append(resnet_block)
            num_filters *= 2

        self.pooling = tf.keras.layers.AveragePooling2D(pool_size=8)
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(self.config["NUM_CLASSES"],
                    activation='softmax',
                    kernel_initializer='he_normal')

    def call(self, inputs, is_training):
        x = self.resnet_layer_1(inputs, is_training=is_training)
        for block in self.resnet_blocks:
            x = block(x, is_training=is_training)
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
