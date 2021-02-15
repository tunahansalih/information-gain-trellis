import os

import numpy as np
import tensorflow as tf

from loss.information_gain import information_gain_loss_fn
from nets.model import InformationGainRoutingModel, Routing
from tqdm import tqdm

import wandb

from utils.helpers import routing_method, current_learning_rate, weight_scheduler, reset_metrics

config = dict(
    # Model Parameters
    DATASET="fashion_mnist",
    # DATASET="cifar100",
    NUM_EPOCHS=100,
    BATCH_SIZE=125,
    USE_ROUTING=True,
    USE_RANDOM_ROUTING=False,
    LR_INITIAL=0.01,
    MOMENTUM=0.9,
    DROPOUT_RATE=float(os.environ.get("DROPOUT_RATE", 0.1)),
    NUM_ROUTES_0=int(os.environ.get("NUM_ROUTES_0", 2)),
    NUM_ROUTES_1=int(os.environ.get("NUM_ROUTES_1", 4)),
    ROUTING_O_LOSS_WEIGHT=float(os.environ.get("ROUTING_O_LOSS_WEIGHT", 1)),
    ROUTING_1_LOSS_WEIGHT=float(os.environ.get("ROUTING_1_LOSS_WEIGHT", 1)),
    ROUTING_LOSS_WEIGHT_DECAY=float(os.environ.get("ROUTING_LOSS_WEIGHT_DECAY", 0.1)),
    ROUTING_0_EARLY_STOPPING_STEP=20000,
    ROUTING_1_EARLY_STOPPING_STEP=20000,
    WEIGHT_DECAY_METHOD="TimeBasedDecay",  # "StepDecay", "ExponentialDecay", "EarlyStopping"
    CNN_0=32,
    CNN_1=64,
    CNN_2=128,
    TAU_INITIAL=1,
    TAU_DECAY_RATE=1,
    INFORMATION_GAIN_BALANCE_COEFFICIENT=float(os.environ.get("INFORMATION_GAIN_BALANCE_COEFFICIENT", 5.0)),
    NO_ROUTING_STEPS=int(os.environ.get("NO_ROUTING_STEPS", 0)),
    RANDOM_ROUTING_STEPS=int(os.environ.get("RANDOM_ROUTING_STEPS", 0))
)
wandb.init(project="information-gain-routing-network", entity="information-gain-routing-network", config=config)

if wandb.config["DATASET"] == "fashion_mnist":
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
    train_x = np.expand_dims(train_x, -1)
    test_x = np.expand_dims(test_x, -1)
    input_shape = (28, 28, 1)
    wandb.config["NUM_CLASSES"] = 10
elif wandb.config["DATASET"] == "cifar100":
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar100.load_data()
    input_shape = (32, 32, 3)
    wandb.config["NUM_CLASSES"] = 100
else:
    raise NotImplementedError

train_x = train_x / 255.0
test_x = test_x / 255.0

train_y = tf.keras.utils.to_categorical(train_y, wandb.config["NUM_CLASSES"])
test_y = tf.keras.utils.to_categorical(test_y, wandb.config["NUM_CLASSES"])

dataset_train = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(60000, seed=3333).batch(
    wandb.config["BATCH_SIZE"])
dataset_validation = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(wandb.config["BATCH_SIZE"])

model = InformationGainRoutingModel(wandb.config)

loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.SGD(lr=wandb.config["LR_INITIAL"], momentum=wandb.config["MOMENTUM"], nesterov=True)

tau = wandb.config["TAU_INITIAL"]

metrics = {"Route0": [tf.keras.metrics.MeanTensor() for i in range(wandb.config["NUM_CLASSES"])],
           "Route1": [tf.keras.metrics.MeanTensor() for i in range(wandb.config["NUM_CLASSES"])],
           "Accuracy": tf.keras.metrics.CategoricalAccuracy(),
           "TotalLoss": tf.keras.metrics.Mean(),
           "Routing0Loss": tf.keras.metrics.Mean(),
           "Routing1Loss": tf.keras.metrics.Mean(),
           "ClassificationLoss": tf.keras.metrics.Mean()}

routing_0_loss_weight = wandb.config["ROUTING_O_LOSS_WEIGHT"]
routing_1_loss_weight = wandb.config["ROUTING_1_LOSS_WEIGHT"]

weight_scheduler_0, weight_scheduler_1 = weight_scheduler(wandb.config)

step = 0
for epoch in range(wandb.config["NUM_EPOCHS"]):
    print(f"Epoch {epoch}")

    reset_metrics(metrics)
    pbar = tqdm(dataset_train)

    for i, (x_batch_train, y_batch_train) in enumerate(pbar):
        step += 1

        current_lr = current_learning_rate(step, wandb.config)
        tf.keras.backend.set_value(optimizer.learning_rate, current_lr)

        current_routing = routing_method(step=step, config=wandb.config)

        routing_0_loss = 0
        routing_1_loss = 0
        information_gain_loss_weight_0 = weight_scheduler_0.get_current_value(step)
        information_gain_loss_weight_1 = weight_scheduler_1.get_current_value(step)
        with tf.GradientTape() as tape:
            route_0, route_1, logits = model(x_batch_train, routing=current_routing, is_training=True)
            classification_loss = loss_fn(y_batch_train, logits)

            if current_routing == Routing.INFORMATION_GAIN_ROUTING:
                route_0 = tf.nn.softmax(route_0 / tau, axis=-1)
                route_1 = tf.nn.softmax(route_1 / tau, axis=-1)
                routing_0_loss = information_gain_loss_weight_0 * information_gain_loss_fn(y_batch_train,
                                                                                           route_0,
                                                                                           balance_coefficient=
                                                                                           wandb.config[
                                                                                               "INFORMATION_GAIN_BALANCE_COEFFICIENT"])
                routing_1_loss = information_gain_loss_weight_1 * information_gain_loss_fn(y_batch_train,
                                                                                           route_1,
                                                                                           balance_coefficient=
                                                                                           wandb.config[
                                                                                               "INFORMATION_GAIN_BALANCE_COEFFICIENT"])
            loss_value = classification_loss + routing_0_loss + routing_1_loss
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # Update Metrics

        # Update metrics
        metrics["Accuracy"].update_state(y_batch_train, logits)
        metrics["TotalLoss"].update_state(loss_value)
        metrics["Routing0Loss"].update_state(routing_0_loss)
        metrics["Routing1Loss"].update_state(routing_1_loss)
        metrics["ClassificationLoss"].update_state(classification_loss)

        # Log metrics
        wandb.log({"Training/TotalLoss": metrics["TotalLoss"].result().numpy(),
                   "Training/ClassificationLoss": metrics["ClassificationLoss"].result().numpy(),
                   "Training/Routing_0_Loss": metrics["Routing0Loss"].result().numpy(),
                   "Training/Routing_1_Loss": metrics["Routing1Loss"].result().numpy(),
                   "Training/Routing_0_Loss_Weight": information_gain_loss_weight_0,
                   "Training/Routing_1_Loss_Weight": information_gain_loss_weight_1,
                   "Training/Accuracy": metrics["Accuracy"].result().numpy(),
                   "Training/SoftmaxSmoothing": tau,
                   "Training/LearningRate": current_lr,
                   "Training/Routing": current_routing.value}, step=step)
        pbar.set_description(
            f"Training Accuracy: %{metrics['Accuracy'].result().numpy() * 100:.2f} Loss: {metrics['TotalLoss'].result().numpy():.5f}")

        if step % 2 == 1:
            tau = tau * wandb.config["TAU_DECAY_RATE"]

    # Validation
    if epoch + 1 % 10 == 0:
        reset_metrics(metrics)
        pbar = tqdm(dataset_validation)
        current_routing = routing_method(step=step, config=wandb.config)
        for (x_batch_val, y_batch_val) in pbar:
            route_0, route_1, logits = model(x_batch_val, routing=current_routing, is_training=False)
            if current_routing in [Routing.RANDOM_ROUTING, Routing.INFORMATION_GAIN_ROUTING]:
                route_0 = tf.nn.softmax(route_0, axis=-1)
                route_1 = tf.nn.softmax(route_1, axis=-1)
                y_batch_val_index = tf.argmax(y_batch_val, axis=-1)
                for c, r_0, r_1 in zip(y_batch_val_index, route_0, route_1):
                    metrics["Route0"][c].update_state(r_0)
                    metrics["Route1"][c].update_state(r_1)

            metrics["Accuracy"].update_state(y_batch_val, logits)

            pbar.set_description(
                f"Validation Accuracy: %{metrics['Accuracy'].result().numpy() * 100:.2f}")

        result_log = {}
        for k in ["Route0", "Route1"]:
            for c, metric in enumerate(metrics[k]):
                data = [[path, ratio] for (path, ratio) in enumerate(metric.result().numpy())]
                table = wandb.Table(data=data, columns=["Route", "Ratio"])
                result_log[f"Validation/{k}/Class_{c}"] = wandb.plot.bar(table, "Route", "Ratio",
                                                                         title=f"{k} Ratios For Class {c}")
        result_log["Epoch"] = epoch
        result_log["Validation/Accuracy"] = metrics["Accuracy"].result().numpy()
        wandb.log(result_log, step=step)
