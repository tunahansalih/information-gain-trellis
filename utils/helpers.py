from loss.scheduling import TimeBasedDecay, StepDecay, ExponentialDecay, EarlyStopping
from nets.model import Routing


def routing_method(step, config):
    if config["NO_ROUTING_STEPS"] > 0 and config["NO_ROUTING_STEPS"] >= step:
        return Routing.NO_ROUTING
    elif config["RANDOM_ROUTING_STEPS"] > 0 and config["RANDOM_ROUTING_STEPS"] >= step:
        return Routing.RANDOM_ROUTING
    else:
        return Routing.INFORMATION_GAIN_ROUTING


def current_learning_rate(step, config):
    if step < 15000:
        return config["LR_INITIAL"]
    elif step < 30000:
        return config["LR_INITIAL"] / 2
    elif step < 40000:
        return config["LR_INITIAL"] / 4
    else:
        return config["LR_INITIAL"] / 40


def weight_scheduler(config):
    if config["WEIGHT_DECAY_METHOD"] == "TimeBasedDecay":
        weight_scheduler_0 = TimeBasedDecay(config["ROUTING_O_LOSS_WEIGHT"],
                                            config["ROUTING_LOSS_WEIGHT_DECAY"])
        weight_scheduler_1 = TimeBasedDecay(config["ROUTING_1_LOSS_WEIGHT"],
                                            config["ROUTING_LOSS_WEIGHT_DECAY"])
    elif config["WEIGHT_DECAY_METHOD"] == "StepDecay":
        weight_scheduler_0 = StepDecay(config["ROUTING_O_LOSS_WEIGHT"], config["ROUTING_LOSS_WEIGHT_DECAY"],
                                       50000 // config["BATCH_SIZE"])
        weight_scheduler_1 = StepDecay(config["ROUTING_1_LOSS_WEIGHT"], config["ROUTING_LOSS_WEIGHT_DECAY"],
                                       50000 // config["BATCH_SIZE"])
    elif config["WEIGHT_DECAY_METHOD"] == "ExponentialDecay":
        weight_scheduler_0 = ExponentialDecay(config["ROUTING_O_LOSS_WEIGHT"],
                                              config["ROUTING_LOSS_WEIGHT_DECAY"])
        weight_scheduler_1 = ExponentialDecay(config["ROUTING_1_LOSS_WEIGHT"],
                                              config["ROUTING_LOSS_WEIGHT_DECAY"])
    elif config["WEIGHT_DECAY_METHOD"] == "EarlyStopping":
        weight_scheduler_0 = EarlyStopping(config["ROUTING_O_LOSS_WEIGHT"],
                                           config["ROUTING_EARLY_STOPPING_STEP"])
        weight_scheduler_1 = EarlyStopping(config["ROUTING_1_LOSS_WEIGHT"],
                                           config["ROUTING_EARLY_STOPPING_STEP"])

    return weight_scheduler_0, weight_scheduler_1


def reset_metrics(metrics_dict):
    for metric in ["Accuracy", "TotalLoss", "Routing0Loss", "Routing1Loss", "ClassificationLoss"]:
        metrics_dict[metric].reset_states()
    for metric in ["Route0", "Route1"]:
        for metric_route in metrics_dict[metric]:
            metric_route.reset_states()
