from enum import Enum

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from nets.layers import (
    InformationGainRoutingBlock,
    RandomRoutingBlock,
    ConvolutionalBlock,
    ResNetBlock,
    ResNetLayer
)


class Routing(Enum):
    RANDOM_ROUTING = 0
    INFORMATION_GAIN_ROUTING = 1


class RoutingModel(models.Model):
    def __init__(self, config):
        super(RoutingModel, self).__init__()
        self.config = config

        # initial blocks
        self.F_0 = None

        # Stacks with routing
        self.F_1 = []  # List of N routes
        self.F_2 = []  # List of N routes

        # Final Stack
        self.F_3 = None

        # Routing blocks for
        if self.config["USE_ROUTING"]:
            self.H_0_random = RandomRoutingBlock(self.config["NUM_ROUTES_0"])
            self.H_1_random = RandomRoutingBlock(self.config["NUM_ROUTES_1"])

            self.H_0 = InformationGainRoutingBlock(self.config["NUM_ROUTES_0"])
            self.H_1 = InformationGainRoutingBlock(self.config["NUM_ROUTES_1"])

    def call(self, inputs, routing, temperature=1, training=True):
        x = inputs

        x = self.F_0(x, training=training)

        if self.config["USE_ROUTING"]:
            if routing == Routing.RANDOM_ROUTING:
                routing_0 = self.H_0_random(x)
            elif routing == Routing.INFORMATION_GAIN_ROUTING:
                routing_0 = self.H_0(x) / temperature
            x = self.apply_routing(x, self.F_1, routing_0, training)
        else:
            x = self.F_1(x, training)

        if self.config["USE_ROUTING"]:
            if routing == Routing.RANDOM_ROUTING:
                routing_1 = self.H_1_random(x)
            elif routing == Routing.INFORMATION_GAIN_ROUTING:
                routing_1 = self.H_1(x) / temperature
            x = self.apply_routing(x, self.F_2, routing_1, training)
        else:
            x = self.F_2(x, training)

        x = self.F_3(x, training)

        return routing_0, routing_1, x

    def apply_routing(self, x, block, routing, training):
        x_output = 0
        for route in range(len(block)):
            selection = tf.where(tf.argmax(routing, axis=-1) == route)
            x_routed = tf.gather_nd(x, selection)
            x_routed = block[route](x_routed, training=training)
            x_routed_shape = tf.shape(x_routed, out_type=tf.dtypes.int64)
            if len(x_routed_shape) == 2:  # Fully connected output
                x_output += tf.scatter_nd(
                    selection,
                    x_routed,
                    tf.stack(
                        [
                            tf.shape(x, out_type=tf.dtypes.int64)[0],
                            x_routed_shape[1]
                        ]
                    ),
                )
            else:  # Convolution output
                x_output += tf.scatter_nd(
                    selection,
                    x_routed,
                    tf.stack(
                        [
                            tf.shape(x, out_type=tf.dtypes.int64)[0],
                            x_routed_shape[1],
                            x_routed_shape[2],
                            x_routed_shape[3],
                        ]
                    ),
                )
        return x_output


class InformationGainRoutingLeNetModel(RoutingModel):
    def __init__(self, config, slim=False):
        super(InformationGainRoutingLeNetModel, self).__init__(config)

        self.config = config

        self.F_0 = tf.keras.Sequential([ConvolutionalBlock(
            filters=32,
            kernel_size=(5, 5),
            padding="same",
        )])

        if config["USE_ROUTING"]:
            for i in range(config["NUM_ROUTES_0"]):
                block = tf.keras.Sequential(layers=[
                    ConvolutionalBlock(
                        filters=64 // config["NUM_ROUTES_0"],
                        kernel_size=(5, 5),
                        padding="same",
                    )
                ])
                self.F_1.append(block)
        else:
            self.F_1 = tf.keras.Sequential(layers=[
                ConvolutionalBlock(
                    filters=64 if not slim else 64 // config["NUM_ROUTES_0"],
                    kernel_size=(5, 5),
                    padding="same",
                )
            ])

        if config["USE_ROUTING"]:
            for i in range(config["NUM_ROUTES_1"]):
                block = tf.keras.Sequential([layers.Flatten(),
                                             layers.Dense(
                                                 units=100 // config["NUM_ROUTES_1"],
                                                 activation=tf.nn.relu,
                                                 kernel_regularizer=regularizers.l2(0.01)
                                             )
                                             ])
                self.F_2.append(block)
        else:
            self.F_2 = tf.keras.Sequential([layers.Flatten(),
                                            layers.Dense(
                                                units=100 if not slim else 100 // config["NUM_ROUTES_1"],
                                                activation=tf.nn.relu,
                                                kernel_regularizer=regularizers.l2(0.01)
                                            )
                                            ])

        self.F_3 = tf.keras.Sequential([layers.Dense(
            units=100,
            activation=tf.nn.relu,
            kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(config["NUM_CLASSES"])
        ])


class InformationGainRoutingResNetModel(RoutingModel):
    def __init__(self, config, slim=False, resnet_depth=18):
        super().__init__()
        self.depth = resnet_depth
        self.config = config
        if (self.depth - 2) % 6 != 0:
            raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")

        num_filters = 16
        num_res_blocks = int((self.depth - 2) / 6)

        # Stack 0
        F_0_layers = [ResNetLayer()]
        for res_block in range(num_res_blocks):
            resnet_block = ResNetBlock(
                stack=0, res_block=res_block, num_filters=num_filters
            )
            F_0_layers.append(resnet_block)

        self.F_0 = tf.keras.Sequential(F_0_layers)

        num_filters *= 2

        # Stack 1 and Routing 0
        if config["USE_ROUTING"]:
            for _ in range(self.config["NUM_ROUTES_0"]):
                resnet_blocks = []
                for res_block in range(num_res_blocks):
                    resnet_blocks.append(
                        ResNetBlock(
                            stack=1,
                            res_block=res_block,
                            num_filters=(num_filters // self.config["NUM_ROUTES_0"]),
                        )
                    )
                self.F_1.append(tf.keras.Sequential(resnet_blocks))
        else:
            resnet_blocks = []
            for res_block in range(num_res_blocks):
                resnet_blocks.append(
                    ResNetBlock(
                        stack=1,
                        res_block=res_block,
                        num_filters=num_filters if not slim else (num_filters // self.config["NUM_ROUTES_0"]),
                    )
                )
            self.F_1 = tf.keras.Sequential(resnet_blocks)

        num_filters *= 2

        # Stack 2 and Routing 1
        if config["USE_ROUTING"]:
            for _ in range(self.config["NUM_ROUTES_0"]):
                resnet_blocks = []
                for res_block in range(num_res_blocks):
                    resnet_blocks.append(
                        ResNetBlock(
                            stack=2,
                            res_block=res_block,
                            num_filters=(num_filters // self.config["NUM_ROUTES_1"]),
                        )
                    )
                self.F_2.append(tf.keras.Sequential(resnet_blocks))
            else:
                resnet_blocks = []
                for res_block in range(num_res_blocks):
                    resnet_blocks.append(
                        ResNetBlock(
                            stack=2,
                            res_block=res_block,
                            num_filters=num_filters if not slim else (num_filters // self.config["NUM_ROUTES_1"]),
                        )
                    )
                self.F_2 = tf.keras.Sequential(resnet_blocks)

        self.F_3 = tf.keras.Sequential([
            layers.AveragePooling2D(pool_size=8),
            layers.Flatten(),
            layers.Dense(
                self.config["NUM_CLASSES"],
                activation=None,
                kernel_initializer="he_normal",
            )
        ])

    @tf.function
    def apply_block(self, x, blocks, training):
        for block in blocks:
            x = block(x, training=training)
        return x

    # @tf.function
    # def apply_routing(self, x, block, routing, training):
    #     x_output = 0
    #     for route in range(len(block)):
    #         selection = tf.where(tf.argmax(routing, axis=-1) == route)
    #         x_routed = tf.gather_nd(x, selection)
    #         x_routed = block[route](x_routed, training=training)
    #         x_routed_shape = tf.shape(x_routed, out_type=tf.dtypes.int64)
    #         x_output += tf.scatter_nd(
    #             selection,
    #             x_routed,
    #             tf.stack(
    #                 [
    #                     tf.shape(x, out_type=tf.dtypes.int64)[0],
    #                     x_routed_shape[1],
    #                     x_routed_shape[2],
    #                     x_routed_shape[3],
    #                 ]
    #                 ),
    #             )
    #         return x_output
