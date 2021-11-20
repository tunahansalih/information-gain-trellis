import os

import numpy as np
import tensorflow as tf
import wandb
from tqdm import tqdm

from nets.model import Routing
from utils.state_helpers import reset_metrics


def validation(model, dataset, name, epoch, config, metrics, global_step):
    reset_metrics(metrics)
    progress_bar = tqdm(dataset, colour='red')
    progress_bar.set_description(f"{name}: Epoch {epoch}")
    if config["USE_ROUTING"]:
        if config["ROUTING_METHOD"] == "Supervised":
            routing = Routing.INFORMATION_GAIN_ROUTING
        elif config["ROUTING_METHOD"] == "Unsupervised":
            routing = Routing.UNSUPERVISED_INFORMATION_GAIN_ROUTING
        else:
            raise NotImplementedError
    else:
        routing = None

    route_0_list = []
    route_1_list = []
    logits_list = []
    y_batch_index_list = []
    for batch, (x_batch, y_batch) in enumerate(progress_bar):
        route_0, route_1, logits = model(
            x_batch, routing=routing, training=False
        )
        y_batch_index = tf.argmax(y_batch, axis=-1)

        route_0_list.append(route_0)
        route_1_list.append(route_1)
        logits_list.append(logits)
        y_batch_index_list.append(y_batch_index)

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
                result_log[f"{name}/{k}/Class_{c}"] = wandb.plot.bar(
                    table, "Route", "Ratio", title=f"{k} Ratios For Class {c}"
                )

    route_0_all = tf.concat(route_0_list, 0)
    route_1_all = tf.concat(route_1_list, 0)
    logits_all = tf.concat(logits_list, 0)
    y_batch_index_all = tf.concat(y_batch_index_list, 0)

    epoch_dir = os.path.join('artifacts', f"{wandb.run.name}-{wandb.run.id}", f"epoch_{epoch}", name)
    os.makedirs(epoch_dir)

    np.savetxt(os.path.join(epoch_dir, 'route_0.csv'), route_0_all, delimiter=',')
    np.savetxt(os.path.join(epoch_dir, 'route_1.csv'), route_1_all, delimiter=',')
    np.savetxt(os.path.join(epoch_dir, 'logit.csv'), logits_all, delimiter=',')
    np.savetxt(os.path.join(epoch_dir, 'y.csv'), y_batch_index_all, delimiter=',')

    result_log[f"{name}/Accuracy"] = metrics["Accuracy"].result().numpy()
    wandb.log(result_log, step=global_step - 1)
