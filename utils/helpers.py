from loss.scheduling import TimeBasedDecay, StepDecay, ExponentialDecay, EarlyStopping
from nets.model import Routing


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
    elif step < 30000:
        return config["LR_INITIAL"] / 2
    elif step < 40000:
        return config["LR_INITIAL"] / 4
    else:
        return config["LR_INITIAL"] / 40


def information_gain_weight_scheduler(config):
    if config["WEIGHT_DECAY_METHOD"] == "TimeBasedDecay":
        weight_scheduler = TimeBasedDecay(
            config["ROUTING_0_LOSS_WEIGHT"], config["ROUTING_LOSS_WEIGHT_DECAY"]
        )

    elif config["WEIGHT_DECAY_METHOD"] == "StepDecay":
        weight_scheduler = StepDecay(
            config["ROUTING_0_LOSS_WEIGHT"],
            config["ROUTING_LOSS_WEIGHT_DECAY"],
            config["NUM_TRAINING"] // config["BATCH_SIZE"] * 10,
        )

    elif config["WEIGHT_DECAY_METHOD"] == "ExponentialDecay":
        weight_scheduler = ExponentialDecay(
            config["ROUTING_0_LOSS_WEIGHT"], config["ROUTING_LOSS_WEIGHT_DECAY"]
        )

    elif config["WEIGHT_DECAY_METHOD"] == "EarlyStopping":
        weight_scheduler = EarlyStopping(
            config["ROUTING_0_LOSS_WEIGHT"], config["ROUTING_EARLY_STOPPING_STEP"]
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
    for metric in ["Route0", "Route1"]:
        for metric_route in metrics_dict[metric]:
            metric_route.reset_states()
