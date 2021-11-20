import tensorflow as tf
import wandb
from tqdm import tqdm

from loss.scheduling import TimeBasedDecay, StepDecay, ExponentialDecay, EarlyStopping
from nets.model import Routing


def validation(model, dataset, name, epoch, config, metrics, global_step):
    reset_metrics(metrics)
    progress_bar = tqdm(dataset, colour='red')
    progress_bar.set_description(f"{name}: Epoch {epoch}")
    for (x_batch, y_batch) in progress_bar:
        route_0, route_1, logits = model(
            x_batch, routing=Routing.INFORMATION_GAIN_ROUTING, training=False
        )
        y_batch_index = tf.argmax(y_batch, axis=-1)
        y_pred_batch_index = tf.argmax(logits, axis=-1)
        if config["USE_ROUTING"]:
            route_0 = tf.nn.softmax(route_0, axis=-1)
            route_1 = tf.nn.softmax(route_1, axis=-1)

            for c, r_0, r_1 in zip(y_batch_index, route_0, route_1):
                metrics["Route0Prob"][c].update_state(r_0)
                metrics["Route1Prob"][c].update_state(r_1)
                metrics["Route0"][c].update_state(tf.math.round(r_0))
                metrics["Route1"][c].update_state(tf.math.round(r_1))

        metrics["Accuracy"].update_state(y_batch_index, y_pred_batch_index)

        progress_bar.set_postfix(
            Accuracy=f"%{metrics['Accuracy'].result().numpy() * 100:.2f}"
        )

    result_log = {}
    if config["USE_ROUTING"]:
        for k in ["Route0", "Route1"]:
            for c, metric in enumerate(metrics[k]):
                data = [
                    [path, ratio]
                    for (path, ratio) in enumerate(metric.result().numpy())
                ]
                table = wandb.Table(data=data, columns=["Route", "Ratio"])
                result_log[f"{name}/{k}/Class_{c}"] = wandb.plot.bar(
                    table, "Route", "Ratio", title=f"{k} Ratios For Class {c}"
                )
        for k in ["Route0Prob", "Route1Prob"]:
            for c, metric in enumerate(metrics[k]):
                data = [
                    [path, ratio]
                    for (path, ratio) in enumerate(metric.result().numpy())
                ]
                table = wandb.Table(data=data, columns=["Route", "Confidence"])
                result_log[f"{name}/{k}/Class_{c}_confidence"] = wandb.plot.bar(
                    table, "Route", "Ratio", title=f"{k} Ratios For Class {c}"
                )
    result_log[f"{name}/Accuracy"] = metrics["Accuracy"].result().numpy()
    wandb.log(result_log, step=global_step - 1)


def routing_method(step, config):
    if config["USE_ROUTING"]:
        if 0 < config["RANDOM_ROUTING_STEPS"] and step < config["RANDOM_ROUTING_STEPS"]:
            return Routing.RANDOM_ROUTING
        else:
            return Routing.INFORMATION_GAIN_ROUTING
    else:
        return None


def current_learning_rate(step, config):
    if step < 30 * config["NUM_TRAINING"] / config["BATCH_SIZE"]:
        return config["LR_INITIAL"]
    elif step < 60 * config["NUM_TRAINING"] / config["BATCH_SIZE"]:
        return config["LR_INITIAL"] / 2
    elif step < 90 * config["NUM_TRAINING"] / config["BATCH_SIZE"]:
        return config["LR_INITIAL"] / 4
    else:
        return config["LR_INITIAL"] / 40


def information_gain_weight_scheduler(config):
    if config["WEIGHT_DECAY_METHOD"] == "TimeBasedDecay":
        weight_scheduler = TimeBasedDecay(
            config["ROUTING_LOSS_WEIGHT"], config["ROUTING_LOSS_WEIGHT_DECAY"]
        )

    elif config["WEIGHT_DECAY_METHOD"] == "StepDecay":
        weight_scheduler = StepDecay(
            config["ROUTING_LOSS_WEIGHT"],
            config["ROUTING_LOSS_WEIGHT_DECAY"],
            config["NUM_TRAINING"] // config["BATCH_SIZE"] * 10,
        )

    elif config["WEIGHT_DECAY_METHOD"] == "ExponentialDecay":
        weight_scheduler = ExponentialDecay(
            config["ROUTING_LOSS_WEIGHT"], config["ROUTING_LOSS_WEIGHT_DECAY"]
        )

    elif config["WEIGHT_DECAY_METHOD"] == "EarlyStopping":
        weight_scheduler = EarlyStopping(
            config["ROUTING_LOSS_WEIGHT"], config["ROUTING_EARLY_STOPPING_STEP"]
        )

    return weight_scheduler


def reset_metrics(metrics_dict):
    for metric in [
        "Accuracy",
        "TotalLoss",
        "Routing0Loss",
        "Routing1Loss",
        "ClassificationLoss",
    ]:
        metrics_dict[metric].reset_states()
    for metric in ["Route0", "Route1", "Route0Prob", "Route1Prob"]:
        for metric_route in metrics_dict[metric]:
            metric_route.reset_states()
