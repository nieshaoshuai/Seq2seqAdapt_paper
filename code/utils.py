import tensorflow as tf


def set_indices(idx, indices):
    indices.append(idx)
    return indices


def keep_indices(idx, indices):
    return indices


def learning_rate_decay(config, global_step):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(config.warmup_steps)
    global_step = tf.to_float(global_step)
    return config.hidden_units ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)


