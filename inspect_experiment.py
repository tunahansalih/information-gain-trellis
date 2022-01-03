import os

import numpy as np

experiment_name = "deft-wind-14-2d4gsyej"
epoch = 99

artifact_src = os.path.join("artifacts", experiment_name, f"epoch_{epoch}")


def softmax(x):
    max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


def accuracy(mode):
    print(mode)
    artifact_dir = os.path.join(artifact_src, mode)
    if os.path.exists(artifact_dir):
        logit = np.loadtxt(os.path.join(artifact_dir, "logit.csv"), delimiter=",")
        y = np.loadtxt(os.path.join(artifact_dir, "y.csv"), dtype=int, delimiter=",")
        print(f"{mode} accuracy: {np.mean(np.argmax(logit, axis=-1) == y) * 100:.2f}%")


def routing_stats(mode):
    print(mode)
    artifact_dir = os.path.join(artifact_src, mode)
    if os.path.exists(artifact_dir):
        route_0 = np.loadtxt(os.path.join(artifact_dir, "route_0.csv"), delimiter=",")
        route_1 = np.loadtxt(os.path.join(artifact_dir, "route_1.csv"), delimiter=",")
        y = np.loadtxt(os.path.join(artifact_dir, "y.csv"), dtype=int, delimiter=",")

        for c in np.unique(y):
            c_route_0 = route_0[y == c]
            c_route_0_confidence = np.mean(softmax(c_route_0), axis=0)
            c_route_0_prob = np.mean(np.round(softmax(c_route_0)), axis=0)
            print(f"Route 0 Class {c} Mean Confidence: {c_route_0_confidence[0]:.2f}% {c_route_0_confidence[1]:.2f}%")
            print(f"Route 0 Class {c} Mean Probability: {c_route_0_prob[0]:.2f}% {c_route_0_prob[1]:.2f}%")

        for c in np.unique(y):
            c_route_1 = route_1[y == c]
            c_route_1_confidence = np.mean(softmax(c_route_1), axis=0)
            c_route_1_prob = np.mean(np.round(softmax(c_route_1)), axis=0)
            print(f"Route 1 Class {c} Mean Confidence: {c_route_1_confidence[0]:.2f}% {c_route_1_confidence[1]:.2f}%")
            print(f"Route 1 Class {c} Mean Probability: {c_route_1_prob[0]:.2f}% {c_route_1_prob[1]:.2f}%")


accuracy("Training")
accuracy("Validation")
accuracy("Test")

routing_stats("Training")
routing_stats("Validation")
routing_stats("Test")
