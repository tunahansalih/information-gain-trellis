import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


def get_dataset(config):
    if config["DATASET"] == "fashion_mnist":
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()
        train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.1,
                                                                        stratify=train_y)
        train_x = np.expand_dims(train_x, -1)
        validation_x = np.expand_dims(validation_x, -1)
        test_x = np.expand_dims(test_x, -1)
        config["NUM_CLASSES"] = 10
        config["INPUT_H"] = 28
        config["INPUT_W"] = 28
        config["INPUT_C"] = 1
    elif config["DATASET"] == "cifar100":
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar100.load_data()
        train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.1,
                                                                        stratify=train_y)
        config["NUM_CLASSES"] = 100
        config["INPUT_H"] = 32
        config["INPUT_W"] = 32
        config["INPUT_C"] = 3
    elif config["DATASET"] == "cifar10":
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
        train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.1,
                                                                        stratify=train_y)
        config["NUM_CLASSES"] = 10
        config["INPUT_H"] = 32
        config["INPUT_W"] = 32
        config["INPUT_C"] = 3

    else:
        raise NotImplementedError

    config["NUM_TRAINING"] = len(train_x)
    config["NUM_VALIDATION"] = len(validation_x)
    config["NUM_TEST"] = len(test_x)
    train_x = train_x / 255.0
    validation_x = validation_x / 255.0
    test_x = test_x / 255.0

    train_y = tf.keras.utils.to_categorical(train_y.flatten(), config["NUM_CLASSES"])
    validation_y = tf.keras.utils.to_categorical(validation_y.flatten(), config["NUM_CLASSES"])
    test_y = tf.keras.utils.to_categorical(test_y.flatten(), config["NUM_CLASSES"])

    dataset_train = (
        tf.data.Dataset.from_tensor_slices((train_x, train_y))
            .shuffle(100000)
            .batch(config["BATCH_SIZE"])
    )
    dataset_validation = tf.data.Dataset.from_tensor_slices((validation_x, validation_y)).batch(
        config["BATCH_SIZE"]
    )

    dataset_test = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(
        config["BATCH_SIZE"]
    )
    return dataset_train, dataset_validation, dataset_test
