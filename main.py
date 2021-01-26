import os

import numpy as np
import tensorflow as tf

from loss.information_gain import information_gain_loss_fn
from nets.routing_layer import InformationGainRoutingBlock, RandomRoutingBlock
from tqdm import tqdm
# from sklearn.model_selection import train_test_split

import wandb

config = dict(
    # Model Parameters
    NUM_EPOCHS=100,
    BATCH_SIZE=125,
    USE_ROUTING=True,
    USE_RANDOM_ROUTING=False,
    LR_INITIAL=0.01,
    DROPOUT_RATE=float(os.environ.get("DROPOUT_RATE", 0.1)),
    NUM_ROUTES_0=int(os.environ.get("NUM_ROUTES_0", 2)),
    NUM_ROUTES_1=int(os.environ.get("NUM_ROUTES_1", 4))
)
wandb.init(project="information-gain-routing-network", entity="information-gain-routing-network", config=config)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
train_x = np.expand_dims(train_x, -1)
test_x = np.expand_dims(test_x, -1)

train_x = train_x / 255.0
test_x = test_x / 255.0

train_y = tf.keras.utils.to_categorical(train_y, 10)
test_y = tf.keras.utils.to_categorical(test_y, 10)

# train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.1)

dataset_train = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(60000, seed=5361).batch(
    config["BATCH_SIZE"])
dataset_validation = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(config["BATCH_SIZE"])
dataset_test = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(config["BATCH_SIZE"])

### Model Definition
input_img = tf.keras.layers.Input((28, 28, 1))
x = tf.keras.layers.Conv2D(32, (5, 5), padding="same")(input_img)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D((2, 2))(x)

x = tf.keras.layers.Conv2D(64, (5, 5), padding="same")(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D((2, 2))(x)

if config["USE_ROUTING"]:
    x, routing_0 = InformationGainRoutingBlock(routes=config["NUM_ROUTES_0"], dropout_rate=config["DROPOUT_RATE"])(x)
elif config["USE_RANDOM_ROUTING"]:
    x, routing_0 = RandomRoutingBlock(routes=config["NUM_ROUTES_0"])(x)

x = tf.keras.layers.Conv2D(128, (5, 5), padding="same")(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D((2, 2))(x)

if config["USE_ROUTING"]:
    x, routing_1 = InformationGainRoutingBlock(routes=config["NUM_ROUTES_1"], dropout_rate=config["DROPOUT_RATE"])(x)
elif config["USE_RANDOM_ROUTING"]:
    x, routing_1 = RandomRoutingBlock(routes=config["NUM_ROUTES_1"])(x)

x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dense(1024)(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(config["DROPOUT_RATE"])(x)
x = tf.keras.layers.Dense(512)(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(config["DROPOUT_RATE"])(x)
x = tf.keras.layers.Dense(10)(x)

if config["USE_ROUTING"] or config["USE_RANDOM_ROUTING"]:
    model = tf.keras.models.Model(input_img, [routing_0, routing_1, x])
else:
    model = tf.keras.models.Model(input_img, x)

loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.SGD(lr=config["LR_INITIAL"], momentum=0.9)

avg_accuracy = tf.keras.metrics.CategoricalAccuracy()
avg_loss = tf.keras.metrics.Mean()

tau = 25

for epoch in range(config["NUM_EPOCHS"]):
    print(f"Epoch {epoch}")
    avg_accuracy.reset_states()
    avg_loss.reset_states()
    pbar = tqdm(dataset_train)

    for i, (x_batch_train, y_batch_train) in enumerate(pbar):
        step = epoch * (len(dataset_train)) + i
        if step == 15000:
            tf.keras.backend.set_value(optimizer.learning_rate, config["LR_INITIAL"] / 2)
        if step == 30000:
            tf.keras.backend.set_value(optimizer.learning_rate, config["LR_INITIAL"] / 4)
        if step == 40000:
            tf.keras.backend.set_value(optimizer.learning_rate, config["LR_INITIAL"] / 40)
        with tf.GradientTape() as tape:
            if config["USE_ROUTING"]:
                route_0, route_1, logits = model(x_batch_train, training=True)
                route_0 = tf.nn.softmax(route_0 / tau, axis=-1)
                route_1 = tf.nn.softmax(route_1 / tau, axis=-1)
                loss_value = loss_fn(y_batch_train, logits)
                loss_value += information_gain_loss_fn(y_batch_train, route_0, balance_coefficient=5.0)
                loss_value += information_gain_loss_fn(y_batch_train, route_1, balance_coefficient=5.0)
            else:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        avg_accuracy.update_state(y_batch_train, logits)
        avg_loss.update_state(loss_value)
        pbar.set_description(
            f"Training Accuracy: %{avg_accuracy.result().numpy() * 100:.2f} Loss: {avg_loss.result().numpy():.5f}")
        wandb.log({"Training/Loss": avg_loss.result().numpy(),
                   "Training/Accuracy": avg_accuracy.result().numpy(),
                   "Training/SoftmaxSmoothing": tau}, step=step)
        if step % 2 == 1:
            tau = tau * 0.9999

    avg_accuracy.reset_states()
    pbar = tqdm(dataset_validation)
    for (x_batch_val, y_batch_val) in pbar:
        if config["USE_ROUTING"] or config["USE_RANDOM_ROUTING"]:
            route_0, route_1, logits = model(x_batch_val, training=False)
        else:
            logits = model(x_batch_val, training=False)
        avg_accuracy.update_state(y_batch_val, logits)

        pbar.set_description(
            f"Validation Accuracy: %{avg_accuracy.result().numpy() * 100:.2f}")
    wandb.log({"Epoch": epoch, "Validation/Accuracy": avg_accuracy.result().numpy()}, step=step)

avg_accuracy.reset_states()
pbar = tqdm(dataset_test)

for (x_batch_test, y_batch_test) in pbar:
    if config["USE_ROUTING"] or config["USE_RANDOM_ROUTING"]:
        route_0, route_1, logits = model(x_batch_test, training=False)
    else:
        logits = model(x_batch_test, training=False)
    avg_accuracy.update_state(y_batch_test, logits)

    pbar.set_description(
        f"Test Accuracy: %{avg_accuracy.result().numpy() * 100:.2f}")

wandb.log({"Test/Accuracy": avg_accuracy.result().numpy()})
