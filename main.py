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


def entropy(prob_distribution):
    log_prob = tf.math.log(prob_distribution + tf.keras.backend.epsilon())

    prob_log_prob = prob_distribution * log_prob
    entropy_val = -1.0 * tf.reduce_sum(prob_log_prob)
    return entropy_val, log_prob


def loss_fn(p_n_given_x_2d, p_c_given_x_2d, balance_coefficient):
    p_n_given_x_3d = tf.expand_dims(input=p_n_given_x_2d, axis=1)
    p_c_given_x_3d = tf.expand_dims(input=p_c_given_x_2d, axis=2)
    non_normalized_joint_xcn = p_n_given_x_3d * p_c_given_x_3d
    # Calculate p(c,n)
    marginal_p_cn = tf.reduce_mean(non_normalized_joint_xcn, axis=0)
    # Calculate p(n)
    marginal_p_n = tf.reduce_sum(marginal_p_cn, axis=0)
    # Calculate p(c)
    marginal_p_c = tf.reduce_sum(marginal_p_cn, axis=1)
    # Calculate entropies
    entropy_p_cn, log_prob_p_cn = entropy(prob_distribution=marginal_p_cn)
    entropy_p_n, log_prob_p_n = entropy(prob_distribution=marginal_p_n)
    entropy_p_c, log_prob_p_c = entropy(prob_distribution=marginal_p_c)
    # Calculate the information gain
    information_gain = (balance_coefficient * entropy_p_n) + entropy_p_c - entropy_p_cn
    information_gain = -1.0 * information_gain
    return information_gain


softmax_decay = 1000
optimizer = tf.optimizers.RMSprop()
accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
losses = []
pbar = tqdm(range(10000))
for step in pbar:
    indices = np.random.randint(len(train_images), size=30)
    x = train_images[indices]
    x = tf.image.resize(x, [224, 224], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
                        antialias=False, name=None)

    y = train_labels[indices].ravel()

    with tf.GradientTape() as tape:
        (output,
         routing_0, mask_0, x_masked_0,
         routing_1, mask_1, x_masked_1,
         routing_2, mask_2, x_masked_2) = model(x, training=True)

        routing_0 = routing_0 / softmax_decay
        routing_0 = tf.nn.softmax(routing_0)
        routing_1 = routing_1 / softmax_decay
        routing_1 = tf.nn.softmax(routing_1)
        routing_2 = routing_2 / softmax_decay
        routing_2 = tf.nn.softmax(routing_2)

        y_one_hot = tf.one_hot(y, depth=100, on_value=1.0, off_value=0.0)

        loss_value = loss_fn(p_n_given_x_2d=routing_0, p_c_given_x_2d=y_one_hot, balance_coefficient=1.0)
        loss_value += loss_fn(p_n_given_x_2d=routing_1, p_c_given_x_2d=y_one_hot, balance_coefficient=1.0)
        loss_value += loss_fn(p_n_given_x_2d=routing_2, p_c_given_x_2d=y_one_hot, balance_coefficient=1.0)
        loss_value += tf.reduce_mean(tf.losses.categorical_crossentropy(y_one_hot, output))
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    accuracy_metric.update_state(y_one_hot, output)
    losses.append(loss_value.numpy())

    if step != 0 and step % 500 == 0:
        accuracy_metric.reset_states()

    softmax_decay *= 0.9

    pbar.set_description(f"Accuracy: %{accuracy_metric.result().numpy() * 100:.2f} Loss: {loss_value.numpy():.5f}")
