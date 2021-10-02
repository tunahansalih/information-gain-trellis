import tensorflow as tf


def entropy(prob_distribution):
    log_prob = tf.math.log(prob_distribution + tf.keras.backend.epsilon())
    prob_log_prob = prob_distribution * log_prob
    entropy_val = -1.0 * tf.reduce_sum(prob_log_prob)
    return entropy_val


def information_gain_loss_fn(p_c_given_x_2d, p_n_given_x_2d, balance_coefficient=1.0):
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
    entropy_p_cn = entropy(prob_distribution=marginal_p_cn)
    entropy_p_n = entropy(prob_distribution=marginal_p_n)
    entropy_p_c = entropy(prob_distribution=marginal_p_c)

    # Calculate the information gain
    information_gain = (balance_coefficient * entropy_p_n) + entropy_p_c - entropy_p_cn

    return -information_gain
