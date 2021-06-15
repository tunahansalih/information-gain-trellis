import numpy as np
import tensorflow as tf

from loss.information_gain import InformationGainLoss
from loss.scheduling import StepDecay
from nets.model import InformationGainRoutingModel, InformationGainRoutingResNetModel, Routing
from tqdm import tqdm

import wandb

from utils.helpers import (
    routing_method,
    current_learning_rate,
    weight_scheduler,
    reset_metrics,
)

wandb.init(
    project='information-gain-routing-network-cifar10',
    entity='tunahansalih',
    config="config.yaml",
)
print(wandb.config)

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
elif wandb.config["DATASET"] == "cifar10":
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
    input_shape = (32, 32, 3)
    wandb.config["NUM_CLASSES"] = 10

else:
    raise NotImplementedError

wandb.config["NUM_TRAINING"] = len(train_x)
wandb.config["NUM_VALIDATION"] = len(test_x)
train_x = train_x / 255.0
test_x = test_x / 255.0

train_y = tf.keras.utils.to_categorical(train_y.flatten(), wandb.config["NUM_CLASSES"])
test_y = tf.keras.utils.to_categorical(test_y.flatten(), wandb.config["NUM_CLASSES"])

dataset_train = (
    tf.data.Dataset.from_tensor_slices((train_x, train_y))
        .shuffle(60000, seed=3333)
        .batch(wandb.config["BATCH_SIZE"])
)
dataset_validation = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(
    wandb.config["BATCH_SIZE"]
)

model = InformationGainRoutingResNetModel(wandb.config)

loss_fn = tf.losses.CategoricalCrossentropy(from_logits=True)

if wandb.config["USE_ROUTING"]:
    information_gain_0_loss_fn = InformationGainLoss(
        num_routes=wandb.config["NUM_ROUTES_0"],
        num_classes=wandb.config["NUM_CLASSES"],
        balance_coefficient=wandb.config["INFORMATION_GAIN_BALANCE_COEFFICIENT"],
        normalize=wandb.config["INFORMATION_GAIN_LOSS_NORMALIZATION"],
    ).loss_fn
    information_gain_1_loss_fn = InformationGainLoss(
        num_routes=wandb.config["NUM_ROUTES_1"],
        num_classes=wandb.config["NUM_CLASSES"],
        balance_coefficient=wandb.config["INFORMATION_GAIN_BALANCE_COEFFICIENT"],
        normalize=wandb.config["INFORMATION_GAIN_LOSS_NORMALIZATION"],
    ).loss_fn
optimizer = tf.optimizers.Adam(
    lr=wandb.config["LR_INITIAL"],
    # momentum=wandb.config["MOMENTUM"],
    # nesterov=True
)

metrics = {
    "Route0": [
        tf.keras.metrics.MeanTensor() for i in range(wandb.config["NUM_CLASSES"])
    ],
    "Route1": [
        tf.keras.metrics.MeanTensor() for i in range(wandb.config["NUM_CLASSES"])
    ],
    "Accuracy": tf.keras.metrics.CategoricalAccuracy(),
    "TotalLoss": tf.keras.metrics.Mean(),
    "Routing0Loss": tf.keras.metrics.Mean(),
    "Routing1Loss": tf.keras.metrics.Mean(),
    "ClassificationLoss": tf.keras.metrics.Mean(),
}

if wandb.config["USE_ROUTING"]:
    weight_scheduler_0, weight_scheduler_1 = weight_scheduler(wandb.config)
    tau_scheduler = StepDecay(
        wandb.config["TAU_INITIAL"], wandb.config["TAU_DECAY_RATE"], decay_step=2
    )

step = 0
for epoch in range(wandb.config["NUM_EPOCHS"]):
    print(f"Epoch {epoch}")

    reset_metrics(metrics)
    progress_bar = tqdm(dataset_train, )

    for i, (x_batch_train, y_batch_train) in enumerate(progress_bar):

        current_lr = current_learning_rate(step, wandb.config)
        tf.keras.backend.set_value(optimizer.learning_rate, current_lr)

        current_routing = routing_method(step=step, config=wandb.config)
        if wandb.config["USE_ROUTING"]:
            information_gain_loss_weight_0 = weight_scheduler_0.get_current_value(step)
            information_gain_loss_weight_1 = weight_scheduler_1.get_current_value(step)
            tau = tau_scheduler.get_current_value(step=step)
        else:
            information_gain_loss_weight_0 = 0
            information_gain_loss_weight_1 = 0
            tau = 1

        with tf.GradientTape(persistent=True) as tape:
            routing_0_loss = 0
            routing_1_loss = 0
            route_0, route_1, logits = model(
                x_batch_train,
                routing=current_routing,
                temperature=tau,
                training=True,
            )
            classification_loss = loss_fn(y_batch_train, logits)

            if (
                    wandb.config["USE_ROUTING"]
                    and current_routing == Routing.INFORMATION_GAIN_ROUTING
            ):
                route_0 = tf.nn.softmax(route_0, axis=-1)
                route_1 = tf.nn.softmax(route_1, axis=-1)
                routing_0_loss = (
                        information_gain_loss_weight_0
                        * information_gain_0_loss_fn(y_batch_train, route_0)
                )
                routing_1_loss = (
                        information_gain_loss_weight_1
                        * information_gain_1_loss_fn(y_batch_train, route_1)
                )
            loss_value = classification_loss + routing_0_loss + routing_1_loss

        if wandb.config["USE_ROUTING"] and wandb.config["DECOUPLE_ROUTING_GRADIENTS"]:
            model_trainable_weights = (
                    model.conv_block_0.trainable_weights
                    + model.batch_norm_0.trainable_weights
                    + model.conv_block_1.trainable_weights
                    + model.batch_norm_1.trainable_weights
                    + model.conv_block_2.trainable_weights
                    + model.batch_norm_2.trainable_weights
                    + model.fc_0.trainable_weights
                    + model.fc_1.trainable_weights
                    + model.fc_2.trainable_weights
            )

            grads = tape.gradient(classification_loss, model_trainable_weights)
            optimizer.apply_gradients(zip(grads, model_trainable_weights))

            if (
                    wandb.config["USE_ROUTING"]
                    and current_routing == Routing.INFORMATION_GAIN_ROUTING
            ):
                grads = tape.gradient(
                    routing_0_loss, model.routing_block_0.trainable_weights
                )
                optimizer.apply_gradients(
                    zip(grads, model.routing_block_0.trainable_weights)
                )

                grads = tape.gradient(
                    routing_1_loss, model.routing_block_1.trainable_weights
                )
                optimizer.apply_gradients(
                    zip(grads, model.routing_block_1.trainable_weights)
                )
        else:
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        del tape
        # Update metrics
        metrics["Accuracy"].update_state(y_batch_train, logits)
        metrics["TotalLoss"].update_state(loss_value)
        metrics["Routing0Loss"].update_state(routing_0_loss)
        metrics["Routing1Loss"].update_state(routing_1_loss)
        metrics["ClassificationLoss"].update_state(classification_loss)

        # Log metrics
        if step % 100 == 0:
            progress_bar.set_description(
                f"Training Accuracy: %{metrics['Accuracy'].result().numpy() * 100:.2f} Loss: {metrics['TotalLoss'].result().numpy():.5f}"
            )

        step += 1

    wandb.log(
        {
            "Training/TotalLoss": metrics["TotalLoss"].result().numpy(),
            "Training/ClassificationLoss": metrics["ClassificationLoss"]
                .result()
                .numpy(),
            "Training/Routing_0_Loss": metrics["Routing0Loss"].result().numpy(),
            "Training/Routing_1_Loss": metrics["Routing1Loss"].result().numpy(),
            "Training/Routing_0_Loss_Weight": information_gain_loss_weight_0,
            "Training/Routing_1_Loss_Weight": information_gain_loss_weight_1,
            "Training/Accuracy": metrics["Accuracy"].result().numpy(),
            "Training/SoftmaxSmoothing": tau,
            "Training/LearningRate": current_lr,
            "Training/Routing": current_routing.value,
        },
        step=step - 1,
    )
    # Validation
    if (epoch + 1) % 10 == 0 or (epoch + 1) == wandb.config["NUM_EPOCHS"]:
        reset_metrics(metrics)
        progress_bar = tqdm(dataset_validation)
        current_routing = routing_method(step=step - 1, config=wandb.config)
        for (x_batch_val, y_batch_val) in progress_bar:
            route_0, route_1, logits = model(
                x_batch_val, routing=current_routing, training=False
            )
            if current_routing in [
                Routing.RANDOM_ROUTING,
                Routing.INFORMATION_GAIN_ROUTING,
            ]:
                route_0 = tf.nn.softmax(route_0, axis=-1)
                route_1 = tf.nn.softmax(route_1, axis=-1)
                y_batch_val_index = tf.argmax(y_batch_val, axis=-1)
                for c, r_0, r_1 in zip(y_batch_val_index, route_0, route_1):
                    metrics["Route0"][c].update_state(r_0)
                    metrics["Route1"][c].update_state(r_1)

            metrics["Accuracy"].update_state(y_batch_val, logits)

            progress_bar.set_description(
                f"Validation Accuracy: %{metrics['Accuracy'].result().numpy() * 100:.2f}"
            )

        result_log = {}
        if wandb.config["USE_ROUTING"]:
            for k in ["Route0", "Route1"]:
                for c, metric in enumerate(metrics[k]):
                    data = [
                        [path, ratio]
                        for (path, ratio) in enumerate(metric.result().numpy())
                    ]
                    table = wandb.Table(data=data, columns=["Route", "Ratio"])
                    result_log[f"Validation/{k}/Class_{c}"] = wandb.plot.bar(
                        table, "Route", "Ratio", title=f"{k} Ratios For Class {c}"
                    )
        result_log["Epoch"] = epoch
        result_log["Validation/Accuracy"] = metrics["Accuracy"].result().numpy()
        wandb.log(result_log, step=step - 1)
