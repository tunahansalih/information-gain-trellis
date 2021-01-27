import tensorflow as tf

from nets.routing import InformationGainRoutingBlock, RandomRoutingBlock


def get_model(input_img, config):
    ### Model Definition
    x = tf.keras.layers.Conv2D(config["CNN_0"], (5, 5), padding="same")(input_img)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(config["CNN_1"], (5, 5), padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    if config["USE_ROUTING"]:
        x, routing_0 = InformationGainRoutingBlock(routes=config["NUM_ROUTES_0"], dropout_rate=config["DROPOUT_RATE"])(
            x)
    elif config["USE_RANDOM_ROUTING"]:
        x, routing_0 = RandomRoutingBlock(routes=config["NUM_ROUTES_0"])(x)

    x = tf.keras.layers.Conv2D(config["CNN_2"], (5, 5), padding="same")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)

    if config["USE_ROUTING"]:
        x, routing_1 = InformationGainRoutingBlock(routes=config["NUM_ROUTES_1"], dropout_rate=config["DROPOUT_RATE"])(
            x)
    elif config["USE_RANDOM_ROUTING"]:
        x, routing_1 = RandomRoutingBlock(routes=config["NUM_ROUTES_1"])(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(1024)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(config["DROPOUT_RATE"])(x)
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(config["DROPOUT_RATE"])(x)
    x = tf.keras.layers.Dense(config["NUM_CLASSES"])(x)

    if config["USE_ROUTING"] or config["USE_RANDOM_ROUTING"]:
        model = tf.keras.models.Model(input_img, [routing_0, routing_1, x])
    else:
        model = tf.keras.models.Model(input_img, x)

    return model
