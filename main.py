import numpy as np
import tensorflow as tf
from nets import resnet50
from tqdm import tqdm

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

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()

train_mean = np.mean(train_images, axis=(0, 1, 2))
train_std = np.std(train_images, axis=(0, 1, 2))

train_images = (train_images - train_mean) / train_std
test_images = (test_images - train_mean) / train_std

model = resnet50.ResNet50(classes=100, backend=tf.keras.backend,
                          layers=tf.keras.layers, models=tf.keras.models, utils=tf.keras.utils)

(output,
 routing_0, mask_0, x_masked_0,
 routing_1, mask_1, x_masked_1,
 routing_2, mask_2, x_masked_2) = model(train_images[:16])
