import numpy as np
import tensorflow as tf


def bivariate_normal_likelihood(x1, x2, mu1, mu2, sigma1, sigma2, rho):
    Z = tf.square(x1 - mu1) / tf.square(sigma1)
    Z += tf.square(x2 - mu2) / tf.square(sigma2)
    Z -= (2 * rho * (x1 - mu1) * (x2 - mu2)) / (sigma1 * sigma2)
    N = tf.exp(-Z / (2 * (1 - tf.square(rho))))
    norm = 2 * np.pi * sigma1 * sigma2
    norm *= tf.sqrt(1 - tf.square(rho))
    N *= (1 / norm)
    return N


def bivariate_normal_sample(mu1, mu2, sigma1, sigma2, rho):
    mean = np.array([mu1, mu2])
    cov_off_diag = rho * sigma1 * sigma2
    cov = np.array([[sigma1**2, cov_off_diag], [cov_off_diag, sigma2**2]])
    return np.random.multivariate_normal(mean, cov, 1)


def unpack_mixture_components(y):
    e = y[:, 0:1]
    pi, mu1, mu2, sigma1, sigma2, rho = tf.split(y[:, 1:], 6, axis=1)

    e = 1 / (1 + tf.exp(e))
    pi = tf.nn.softmax(pi)
    sigma1 = tf.exp(sigma1)
    sigma2 = tf.exp(sigma2)
    rho = tf.tanh(rho)
    return e, pi, mu1, mu2, sigma1, sigma2, rho


def get_output_dim_for_mixture(mixture_components=20):
    """
    for mixture_componenets = 20
    output dimension is (20 weights, 40 means, 40 standard deviations
      and 20 correlations) + 1 end-of-stroke
    output_dim = 121 = 6 * mixture_components + 1
    """
    return mixture_components * 6 + 1


def get_mixture_loss(pi, e, mu1, mu2, sigma1, sigma2, rho, x1, x2, x3):
    """
    Get loss for mixture network
    """

    end_prob = tf.expand_dims(e * x1 + (1 - e) * (1 - x1), 1)
    stroke_prob = stroke_prob = bivariate_normal_likelihood(
        x2, x3, mu1, mu2, sigma1, sigma2, rho)

    eps = np.finfo(np.float32).eps
    loss = -tf.log(
        tf.maximum(tf.reduce_sum(stroke_prob * pi, axis=1), eps))
    loss += -tf.log(tf.maximum(end_prob, eps))
    loss = tf.reduce_mean(loss)

    mse = tf.square(tf.reduce_sum(pi * mu1, axis=1) - x2) +\
        tf.square(tf.reduce_sum(pi * mu2, axis=1) - x3)
    mse = tf.reduce_mean(mse)

    return loss, mse


def build_stroke_mixture_model(model, output, targets, mixture_components,
                               max_grad_norm, learning_rate, decay, momentum):
    """
    Build model on top of last layer of an RNN for the
        stroke gaussian mixture model output
    """
    stroke_dim = 3

    output_dim = get_output_dim_for_mixture(mixture_components)

    # get the mixture parameters
    output = tf.layers.dense(
        output, output_dim, activation=None)
    # output is (num_steps * batch_size, 121)

    x1, x2, x3 = tf.split(
        tf.reshape(targets, [-1, stroke_dim]),
        stroke_dim, axis=1)
    model.x1, model.x2, model.x3 = x1, x2, x3
    # x1/x2/x3 is (num_steps * batch_size, 3)

    # build the mixture model here
    params = unpack_mixture_components(output)
    param_names = ['e', 'pi', 'mu1', 'mu2', 'sigma1', 'sigma2', 'rho']
    for name, p in zip(param_names, params):
        setattr(model, name, p)
    e, pi, mu1, mu2, sigma1, sigma2, rho = params
    model.bivariate_normal_params = params

    model.loss, model.mse = get_mixture_loss(
        pi, e, mu1, mu2, sigma1, sigma2, rho, x1, x2, x3)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(model.loss, tvars), max_grad_norm)
    model.grads = grads
    model.optimizer = tf.train.RMSPropOptimizer(
        learning_rate, decay=decay, momentum=momentum)

    model.train_op = model.optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())


# def rollout_conditional_model(inputs, text_input, cells, initial_states,
#                               char_seq_len, num_layers,
#                               batch_size, window_components, vocab_size,
#                               _train, dropout, hidden_size):
#     """
#     Rollout conditional stroke generation model
#     """


#     states = initial_states

#     last_outputs = []
#     u = np.array([i for i in range(1, char_seq_len + 1)], dtype=np.float32)

#     last_kappa = init_kappa = tf.zeros(
#         dtype=tf.float32, shape=[batch_size, window_components, 1])
#     init_wt = tf.zeros(
#         dtype=tf.float32, shape=[batch_size, vocab_size])
#     last_wt = init_wt

#     with tf.variable_scope('condition-rollout', tf.AUTO_REUSE):
#         for idx, i in enumerate(inputs):
#             # unroll the network to get outputs
#             out_cell_0, states[0] = cells[0](
#                 tf.concat([i, last_wt], axis=1), states[0])

#             # character mixture model
#             char_mixture_components = tf.layers.dense(
#                 out_cell_0, window_components*3, activation=None)
#             alpha, beta, kappa = tf.split(char_mixture_components, 3, axis=1)
#             alpha = tf.expand_dims(tf.exp(alpha), -1)
#             beta = tf.expand_dims(tf.exp(beta), -1)
#             kappa = last_kappa + tf.expand_dims(tf.exp(kappa), -1)
#             last_kappa = kappa

#             # character mixture weights
#             phi = alpha * tf.exp(
#                 -beta * tf.square(kappa - u))
#             phi = tf.reduce_sum(phi, axis=1, keepdims=True)

#             w_t = tf.squeeze(tf.matmul(phi, text_input), axis=1)

#             # remaining layers with skip connections
#             out_cell_1, states[1] = cells[1](
#                 tf.concat([i, out_cell_0, w_t], axis=1), states[1])

#             out_cell_2, states[2] = cells[2](
#                 tf.concat([i, out_cell_0, out_cell_1, w_t], axis=1), states[2])

#             last_outputs.append(out_cell_2)
