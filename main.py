import os

import numpy as np
import tensorflow as tf

from loss.information_gain import information_gain_loss_fn
from nets.routing import InformationGainRoutingBlock, RandomRoutingBlock
from tqdm import tqdm
# from sklearn.model_selection import train_test_split

import wandb

config = dict(
    # Model Parameters
    DATASET="fashion_mnist",
    # DATASET="cifar100",
    NUM_EPOCHS=100,
    BATCH_SIZE=125,
    USE_ROUTING=True,
    USE_RANDOM_ROUTING=False,
    LR_INITIAL=0.01,
    DROPOUT_RATE=float(os.environ.get("DROPOUT_RATE", 0.1)),
    NUM_ROUTES_0=int(os.environ.get("NUM_ROUTES_0", 2)),
    NUM_ROUTES_1=int(os.environ.get("NUM_ROUTES_1", 4)),
    CNN_0=32,
    CNN_1=64,
    CNN_2=128
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

if config["DATASET"] == "fashion_mnist":
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
    train_x = np.expand_dims(train_x, -1)
    test_x = np.expand_dims(test_x, -1)
    input_img = tf.keras.layers.Input((28, 28, 1))
    NUM_CLASSES = 10

elif config["DATASET"] == "cifar100":
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar100.load_data()
    input_img = tf.keras.layers.Input((32, 32, 3))
    NUM_CLASSES = 100

train_x = train_x / 255.0
test_x = test_x / 255.0

train_y = tf.keras.utils.to_categorical(train_y, NUM_CLASSES)
test_y = tf.keras.utils.to_categorical(test_y, NUM_CLASSES)

# train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.1)

dataset_train = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(60000, seed=5361).batch(
    config["BATCH_SIZE"])
dataset_validation = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(config["BATCH_SIZE"])
dataset_test = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(config["BATCH_SIZE"])

### Model Definition
x = tf.keras.layers.Conv2D(config["CNN_0"], (5, 5), padding="same")(input_img)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D((2, 2))(x)

x = tf.keras.layers.Conv2D(config["CNN_1"], (5, 5), padding="same")(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.MaxPool2D((2, 2))(x)

if config["USE_ROUTING"]:
    x, routing_0 = InformationGainRoutingBlock(routes=config["NUM_ROUTES_0"], dropout_rate=config["DROPOUT_RATE"])(x)
elif config["USE_RANDOM_ROUTING"]:
    x, routing_0 = RandomRoutingBlock(routes=config["NUM_ROUTES_0"])(x)

x = tf.keras.layers.Conv2D(config["CNN_2"], (5, 5), padding="same")(x)
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
x = tf.keras.layers.Dense(NUM_CLASSES)(x)

if config["USE_ROUTING"] or config["USE_RANDOM_ROUTING"]:
    model = tf.keras.models.Model(input_img, [routing_0, routing_1, x])
else:
    model = tf.keras.models.Model(input_img, x)

loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.SGD(lr=config["LR_INITIAL"], momentum=0.9)

avg_accuracy = tf.keras.metrics.CategoricalAccuracy()
avg_loss = tf.keras.metrics.Mean()
avg_routing_0_loss = tf.keras.metrics.Mean()
avg_routing_1_loss = tf.keras.metrics.Mean()
avg_classification_loss = tf.keras.metrics.Mean()

tau = 25

avg_route_0_probs = []
avg_route_1_probs = []
for c in range(NUM_CLASSES):
    avg_route_0_probs.append(tf.keras.metrics.MeanTensor())
    avg_route_1_probs.append(tf.keras.metrics.MeanTensor())

for epoch in range(config["NUM_EPOCHS"]):
    print(f"Epoch {epoch}")
    avg_accuracy.reset_states()
    avg_loss.reset_states()
    avg_routing_0_loss.reset_states()
    avg_routing_1_loss.reset_states()
    avg_classification_loss.reset_states()
    pbar = tqdm(dataset_train)

    for i, (x_batch_train, y_batch_train) in enumerate(pbar):
        step = epoch * (len(dataset_train)) + i
        if step == 15000:
            tf.keras.backend.set_value(optimizer.learning_rate, config["LR_INITIAL"] / 2)
        if step == 30000:
            tf.keras.backend.set_value(optimizer.learning_rate, config["LR_INITIAL"] / 4)
        if step == 40000:
            tf.keras.backend.set_value(optimizer.learning_rate, config["LR_INITIAL"] / 40)
        classification_loss = 0
        routing_0_loss = 0
        routing_1_loss = 0
        with tf.GradientTape() as tape:

            if config["USE_ROUTING"]:
                route_0, route_1, logits = model(x_batch_train, training=True)
                route_0 = tf.nn.softmax(route_0 / tau, axis=-1)
                route_1 = tf.nn.softmax(route_1 / tau, axis=-1)
                classification_loss = loss_fn(y_batch_train, logits)
                routing_0_loss = information_gain_loss_fn(y_batch_train, route_0, balance_coefficient=5.0)
                routing_1_loss = information_gain_loss_fn(y_batch_train, route_1, balance_coefficient=5.0)
            elif config["USE_RANDOM_ROUTING"]:
                route_0, route_1, logits = model(x_batch_train, training=True)
                classification_loss = loss_fn(y_batch_train, logits)
            else:
                logits = model(x_batch_train, training=True)
                classification_loss = loss_fn(y_batch_train, logits)
            loss_value = classification_loss + routing_0_loss + routing_1_loss
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # Update Metrics
        avg_accuracy.update_state(y_batch_train, logits)
        avg_loss.update_state(loss_value)
        avg_routing_0_loss.update_state(routing_0_loss)
        avg_routing_1_loss.update_state(routing_1_loss)
        avg_classification_loss.update_state(classification_loss)
        pbar.set_description(
            f"Training Accuracy: %{avg_accuracy.result().numpy() * 100:.2f} Loss: {avg_loss.result().numpy():.5f}")

        # Log metrics
        wandb.log({"Training/Loss": avg_loss.result().numpy(),
                   "Training/ClassificationLoss": avg_classification_loss.result().numpy(),
                   "Training/Routing_0_Loss": avg_routing_0_loss.result().numpy(),
                   "Training/Routing_1_Loss": avg_routing_1_loss.result().numpy(),
                   "Training/Accuracy": avg_accuracy.result().numpy(),
                   "Training/SoftmaxSmoothing": tau,
                   "Training/LearningRate": optimizer.lr.numpy()}, step=step)
        if step % 2 == 1:
            tau = tau * 0.9999

    avg_accuracy.reset_states()
    pbar = tqdm(dataset_validation)
    for c in range(NUM_CLASSES):
        avg_route_0_probs[c].reset_states()
        avg_route_1_probs[c].reset_states()
    for (x_batch_val, y_batch_val) in pbar:
        if config["USE_ROUTING"] or config["USE_RANDOM_ROUTING"]:
            route_0, route_1, logits = model(x_batch_val, training=False)
            route_0 = tf.nn.softmax(route_0, axis=-1)
            route_1 = tf.nn.softmax(route_1, axis=-1)
            for y, r_0, r_1 in zip(y_batch_val, route_0, route_1):
                c = np.argmax(y)
                avg_route_0_probs[c].update_state(r_0)
                avg_route_1_probs[c].update_state(r_1)
        else:
            logits = model(x_batch_val, training=False)
        avg_accuracy.update_state(y_batch_val, logits)

        pbar.set_description(
            f"Validation Accuracy: %{avg_accuracy.result().numpy() * 100:.2f}")
    result_log = {}
    for c, route_0_avg, route_1_avg in zip(range(NUM_CLASSES), avg_route_0_probs, avg_route_1_probs):
        data = [[label, val] for (label, val) in enumerate(avg_route_0_probs[c].result().numpy())]
        table = wandb.Table(data=data, columns=["route", "prob"])
        result_log[f"Validation/Route_0/Class_{c}"] = wandb.plot.bar(table, "route", "prob",
                                                                     title=f"Route 0 Probabilities For Class {c}")

        data = [[label, val] for (label, val) in enumerate(avg_route_1_probs[c].result().numpy())]
        table = wandb.Table(data=data, columns=["route", "prob"])
        result_log[f"Validation/Route_1/Class_{c}"] = wandb.plot.bar(table, "route", "prob",
                                                                     title=f"Route 1 Probabilities For Class {c}")

    result_log["Epoch"] = epoch
    result_log["Validation/Accuracy"] = avg_accuracy.result().numpy()
    wandb.log(result_log, step=step)

avg_accuracy.reset_states()
pbar = tqdm(dataset_validation)
for c in range(NUM_CLASSES):
    avg_route_0_probs[c].reset_states()
avg_route_1_probs[c].reset_states()

for (x_batch_test, y_batch_test) in pbar:
    if config["USE_ROUTING"] or config["USE_RANDOM_ROUTING"]:
        route_0, route_1, logits = model(x_batch_test, training=False)
        route_0 = tf.nn.softmax(route_0)
        route_1 = tf.nn.softmax(route_1)
        for y_batch in zip(y_batch_test, route_0, route_1):
            c = np.argmax(y_batch)
        avg_route_0_probs[c].update_state(route_0)
        avg_route_1_probs[c].update_state(route_1)
    else:
        logits = model(x_batch_test, training=False)
        avg_accuracy.update_state(y_batch_test, logits)

pbar.set_description(
    f"Test Accuracy: %{avg_accuracy.result().numpy() * 100:.2f}")
result_log = {}
for c, route_0_avg, route_1_avg in zip(range(NUM_CLASSES), avg_route_0_probs, avg_route_1_probs):
    data = [[label, val] for (label, val) in
            zip(range(config["NUM_ROUTES_0"]), avg_route_0_probs[c].result().numpy())]
    table = wandb.Table(data=data, columns=["route", "prob"])
    result_log[f"Validation/Route_0/Class_{c}"] = wandb.plot.bar(table, "route", "prob",
                                                                 title="Route Probabilities")

    data = [[label, val] for (label, val) in
            zip(range(config["NUM_ROUTES_0"]), avg_route_1_probs[c].result().numpy())]
    table = wandb.Table(data=data, columns=["route", "prob"])
    result_log[f"Validation/Route_1/Class_{c}"] = wandb.plot.bar(table, "route", "prob",
                                                                 title="Route Probabilities")
result_log["Test/Accuracy"] = avg_accuracy.result().numpy()
wandb.log(result_log, step=step)
