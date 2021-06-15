from enum import Enum
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers

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


class InformationGainRoutingModel(models.Model):
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

        self.flatten = layers.Flatten()

        self.fc_0 = layers.Dense(1024, activation=tf.nn.relu)
        self.do_0 = layers.Dropout(config["DROPOUT_RATE"])
        self.fc_1 = layers.Dense(512, activation=tf.nn.relu)
        self.do_1 = layers.Dropout(config["DROPOUT_RATE"])
        self.fc_2 = layers.Dense(config["NUM_CLASSES"])

    def call(self, inputs, routing: Routing, temperature=1, training=True):
        x = self.conv_block_0(inputs)
        x = self.batch_norm_0(x, training=training)

        if routing == Routing.RANDOM_ROUTING:
            routing_0 = self.random_routing_block_0(x)
        elif routing == Routing.INFORMATION_GAIN_ROUTING:
            routing_0 = self.routing_block_0(x, training=training) / temperature
        elif routing == Routing.NO_ROUTING:
            routing_0 = None
        else:
            routing_0 = None

        x = self.conv_block_1(x)
        if routing_0 is not None:
            x = self.routing_mask_layer_0(x, routing_0, training=training)
        x = self.batch_norm_1(x, training=training)

        if routing == Routing.RANDOM_ROUTING:
            routing_1 = self.random_routing_block_1(x)
        elif routing == Routing.INFORMATION_GAIN_ROUTING:
            routing_1 = self.routing_block_1(x, training=training) / temperature
        elif routing == Routing.NO_ROUTING:
            routing_1 = None
        else:
            routing_1 = None

        x = self.conv_block_2(x)
        if routing_1 is not None:
            x = self.routing_mask_layer_1(x, routing_1, training=training)
        x = self.batch_norm_2(x, training=training)

        x = self.flatten(x)

        x = self.fc_0(x)
        if training:
            x = self.do_0(x)
        x = self.fc_1(x)
        if training:
            x = self.do_1(x)
        x = self.fc_2(x)

        return routing_0, routing_1, x


class InformationGainRoutingLeNetModel(models.Model):
    def __init__(self, config):
        super(InformationGainRoutingLeNetModel, self).__init__()

        self.config = config

        self.conv_block_0 = ConvolutionalBlock(
            filters=config["CNN_0"],
            kernel_size=(5, 5),
            padding="same",
        )

        self.routing_block_0 = InformationGainRoutingBlock(
            routes=config["NUM_ROUTES_0"]
        )
        self.conv_blocks_1 = []
        for i in range(config["NUM_ROUTES_0"]):
            self.conv_blocks_1.append(
                ConvolutionalBlock(
                    filters=config["CNN_1"] // config["NUM_ROUTES_0"],
                    kernel_size=(5, 5),
                    padding="same",
                )
            )

        self.flatten = layers.Flatten()

        self.fc_0 = layers.Dense(
            100,
            activation=tf.nn.relu,
            kernel_regularizer=regularizers.l2(config["L2_WEIGHT_DECAY"]),
        )
        self.fc_1 = layers.Dense(config["NUM_CLASSES"])

    def call(self, inputs, temperature=1, training=True):
        x = self.conv_block_0(inputs)

        routing_0 = self.routing_block_0(x, training=training) / temperature

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


class InformationGainRoutingResNetModel(models.Model):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if (self.config["RESNET_DEPTH"] - 2) % 6 != 0:
            raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
        # Start model definition.

        self.resnet_layer_1 = ResNetLayer()

        num_filters = 16
        num_res_blocks = int((self.config["RESNET_DEPTH"] - 2) / 6)

        self.resnet_blocks = []
        # Stack 0
        self.stack_0_blocks = []
        for res_block in range(num_res_blocks):
            resnet_block = ResNetBlock(
                stack=0, res_block=res_block, num_filters=num_filters
            )
            self.stack_0_blocks.append(resnet_block)
        num_filters *= 2

        # Stack 1 and Routing 0
        self.routing_block_0 = InformationGainRoutingBlock(
            routes=self.config["NUM_ROUTES_0"]
        )
        self.random_routing_block_0 = RandomRoutingBlock(
            routes=config["NUM_ROUTES_0"]
        )
        self.stack_1_blocks = []
        for res_block in range(num_res_blocks):
            if config["USE_ROUTING"]:
                route_blocks = []
                for _ in range(self.config["NUM_ROUTES_0"]):
                    route_blocks.append(
                        ResNetBlock(
                            stack=1,
                            res_block=res_block,
                            num_filters=(num_filters // self.config["NUM_ROUTES_0"]),
                        )
                    )
                self.stack_1_blocks.append(route_blocks)
            else:
                stack_1_block = ResNetBlock(
                    stack=1,
                    res_block=res_block,
                    num_filters=num_filters,
                )
                self.stack_1_blocks.append(stack_1_block)
        num_filters *= 2

        # Stack 2 and Routing 1
        self.routing_block_1 = InformationGainRoutingBlock(
            routes=config["NUM_ROUTES_1"]
        )
        self.random_routing_block_1 = RandomRoutingBlock(
            routes=config["NUM_ROUTES_1"]
        )
        self.stack_2_blocks = []
        for res_block in range(num_res_blocks):
            if config["USE_ROUTING"]:
                route_blocks = []
                for _ in range(self.config["NUM_ROUTES_0"]):
                    route_blocks.append(
                        ResNetBlock(
                            stack=2,
                            res_block=res_block,
                            num_filters=(num_filters // self.config["NUM_ROUTES_1"]),
                        )
                    )
                self.stack_2_blocks.append(route_blocks)
            else:
                stack_2_block = ResNetBlock(
                    stack=2,
                    res_block=res_block,
                    num_filters=num_filters,
                )
                self.stack_2_blocks.append(stack_2_block)

        self.pooling = layers.AveragePooling2D(pool_size=8)
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(
            self.config["NUM_CLASSES"],
            activation=None,
            kernel_initializer="he_normal",
        )

    def call(self, inputs, routing: Routing, temperature=1, training=True):
        x = self.resnet_layer_1(inputs, training=training)
        for block in self.stack_0_blocks:
            x = block(x, training=training)

        if routing == Routing.RANDOM_ROUTING:
            routing_0 = self.random_routing_block_0(x)
        elif routing == Routing.INFORMATION_GAIN_ROUTING:
            routing_0 = self.routing_block_0(x, training=training) / temperature
        elif routing == Routing.NO_ROUTING:
            routing_0 = None
        else:
            routing_0 = None

        if routing is not None:
            x_output = 0
            for route in range(self.config["NUM_ROUTES_0"]):
                selection = tf.where(tf.argmax(routing_0, axis=-1) == route)
                x_routed = tf.gather_nd(x, selection)
                for block in self.stack_1_blocks:
                    x_routed = block[route](x_routed)
                x_routed_shape = tf.shape(x_routed, out_type=tf.dtypes.int64)
                x_output += tf.scatter_nd(selection, x_routed, tf.stack(
                    [tf.shape(x, out_type=tf.dtypes.int64)[0], x_routed_shape[1], x_routed_shape[2],
                     x_routed_shape[3]]))

            x = x_output
        else:
            for block in self.stack_1_blocks:
                x = block(x, training=training)

        if routing == Routing.RANDOM_ROUTING:
            routing_1 = self.random_routing_block_1(x)
        elif routing == Routing.INFORMATION_GAIN_ROUTING:
            routing_1 = self.routing_block_1(x, training=training) / temperature
        elif routing == Routing.NO_ROUTING:
            routing_1 = None
        else:
            routing_1 = None

        if routing is not None:
            x_output = 0
            for route in range(self.config["NUM_ROUTES_1"]):
                selection = tf.where(tf.argmax(routing_1, axis=-1) == route)
                x_routed = tf.gather_nd(x, selection)
                for block in self.stack_2_blocks:
                    x_routed = block[route](x_routed)
                x_routed_shape = tf.shape(x_routed, out_type=tf.dtypes.int64)
                x_output += tf.scatter_nd(selection, x_routed, tf.stack(
                    [tf.shape(x, out_type=tf.dtypes.int64)[0], x_routed_shape[1], x_routed_shape[2],
                     x_routed_shape[3]]))

            x = x_output
        else:
            for block in self.stack_2_blocks:
                x = block(x, training=training)

        x = self.pooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        return routing_0, routing_1, x
