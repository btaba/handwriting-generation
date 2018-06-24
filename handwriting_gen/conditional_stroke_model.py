"""
Conditional stroke generation
"""
import click
import plotting
import numpy as np
import tensorflow as tf
from data_manager import (
    get_preprocessed_data_splits, conditional_batch_generator,
    sent_to_int, pad_vec
)
from networks import TFModel, get_lstm_cells_and_states
from networks import create_tf_session
from distributions import build_stroke_mixture_model
from sample import sample_stroke

STROKE_DIM = 3  # (end-of-stroke, x, y) - input dimension


class ConditionalStrokeModel(TFModel):

    def __init__(self, save_path,
                 learning_rate=1e-4,
                 mixture_components=20,
                 window_components=10,
                 char_seq_len=30,
                 char_dict={},
                 hidden_size=200, dropout=0.0,
                 max_grad_norm=5, decay=0.95, momentum=0.9,
                 rnn_steps=200, batch_size=32, is_train=True, **kwargs):
        """
        :param save_path: str, path to model checkpoints
        :param learning_rate: float
        :param mixture_components: int,
            number of mixture components to predict strokes
        :param window_components: int, number of components for window model
        :param char_seq_len: int, length of character sequence
        :param num_layers: int, number of layers for stacked lstm
        :param hidden_size: int, number of hidden units for lstm
        :param dropout: float, prob for output dropout
        :param max_grad_norm: int, gradient clipping value
        :param decay: float, decay for RMSProp
        :param momentum: float, momentutm for RMSProp
        :param rnn_steps: int, steps to rollout RNN
        :param batch_size: int, size of batch for training
        :param is_train: bool, whether we are training
            or evaluating the network
        """
        super().__init__(save_path)

        self.sess = create_tf_session()

        self.learning_rate = learning_rate
        self.mixture_components = mixture_components

        # there are 3 mixture components for each window component
        self.window_components = window_components
        self.window_output_dim = window_components * 3

        self.vocab_size = len(char_dict)
        self.char_seq_len = char_seq_len
        self.char_dict = char_dict

        self.hidden_size = hidden_size
        self.num_layers = 3
        self.dropout = dropout
        self.max_grad_norm = max_grad_norm
        self.decay = decay
        self.momentum = momentum
        self.rnn_steps = rnn_steps
        self.batch_size = batch_size
        self.is_train = is_train

        self.build()
        self.sess.run(tf.global_variables_initializer())

    def build(self):

        # the input is the stroke sequence
        self.inputs = inputs = tf.placeholder(
            tf.float32, [None, self.rnn_steps, STROKE_DIM], 'input')
        # the target is the next stroke
        self.targets = tf.placeholder(
            tf.float32, [None, self.rnn_steps, STROKE_DIM], 'targets')
        # the input characters for the sequence
        self.input_characters = tf.placeholder(
            tf.int32, [None, self.char_seq_len],
            'characters')
        self.characters = tf.one_hot(
            self.input_characters, depth=self.vocab_size)

        with tf.variable_scope('char-stroke-gen', reuse=tf.AUTO_REUSE):
            # split inputs by rnn steps
            inputs = [tf.squeeze(i, [1]) for i in
                      tf.split(self.inputs, self.rnn_steps, axis=1)]

            cells, initial_states = get_lstm_cells_and_states(
                self.hidden_size, self.is_train, self.num_layers,
                self.batch_size, self.dropout)
            self.initial_states = initial_states
            state_0, state_1, state_2 = initial_states

            self.init_kappa = init_kappa = tf.zeros(
                dtype=tf.float32,
                shape=[self.batch_size, self.window_components, 1])
            self.init_wt = last_wt = self.characters[:, 0, :]
            last_kappa = init_kappa

            last_outputs = []
            u = np.array([i for i in range(1, self.char_seq_len + 1)],
                         dtype=np.float32)

            for idx, i in enumerate(inputs):
                with tf.variable_scope('cells', tf.AUTO_REUSE):
                    # unroll the network to get outputs
                    # out_cell_0, state_0 = cells[0](
                    #     tf.concat([i, last_wt], axis=1), state_0)
                    out_cell_0, state_0 = cells[0](i, state_0)

                    # character mixture model
                    char_mixture_components = tf.layers.dense(
                        out_cell_0, self.window_output_dim, activation=None)
                    alpha, beta, kappa = tf.split(
                        char_mixture_components, 3,
                        axis=1)
                    alpha = tf.expand_dims(tf.exp(alpha), -1)
                    beta = tf.expand_dims(tf.exp(beta), -1)
                    kappa = last_kappa + tf.expand_dims(tf.exp(kappa), -1)
                    last_kappa = kappa

                    # character mixture weights
                    phi = alpha * tf.exp(
                        -beta * tf.square(kappa - u))
                    phi = tf.reduce_sum(phi, axis=1, keepdims=True)
                    self.phi = phi
                    w_t = tf.squeeze(tf.matmul(phi, self.characters), axis=1)

                    # remaining 2 layers with skip connections
                    out_cell_1, state_1 = cells[1](
                        tf.concat([i, out_cell_0, w_t], axis=1), state_1)

                    # remove last layer for speedup in prototyping
                    out_cell_2, state_2 = cells[2](
                        tf.concat([i, out_cell_1, w_t], axis=1),
                        state_2)

                    last_outputs.append(out_cell_2)
                    last_wt = w_t

            self.last_states = [state_0, state_1, state_2]
            self.last_wt = last_wt
            self.last_kappa = last_kappa

            output = tf.reshape(
                tf.concat(last_outputs, 1), [-1, self.hidden_size])

            build_stroke_mixture_model(
                self, output, self.targets, self.mixture_components,
                self.max_grad_norm, self.learning_rate, self.decay,
                self.momentum)

        self.saver = tf.train.Saver()

    def train(self, inputs, targets, chars):
        return self.sess.run(
            [self.loss, self.train_op],
            {self.inputs: inputs, self.targets: targets,
             self.input_characters: chars})


def decode(model, text='Hey I am baruch, nice to meet you',
           char_seq_len=30,
           steps=700, seed=42, std_bias=10, mixture_bias=1):
    """
    Decode strokes from network conditional on text input
    """

    np.random.seed(seed)

    text_int = sent_to_int(list(text), model.char_dict)
    text_int = pad_vec([text_int], char_seq_len)[0]

    feed_dict = {}
    feed_dict[model.input_characters] = [text_int]

    strokes = [np.array([[[1, 0, 0]]])]
    last_state, last_kappa, last_wt = model.sess.run(
        [model.initial_states, model.init_kappa, model.init_wt],
        feed_dict)

    for _ in range(steps):

        feed_dict[model.inputs] = strokes[-1]
        for idx, i in enumerate(model.initial_states):
            feed_dict[i] = last_state[idx]
        feed_dict[model.init_kappa] = last_kappa
        feed_dict[model.init_wt] = last_wt

        last_state, last_wt, last_kappa, (e, pi, *bv_params) = model.sess.run(
            [model.last_states, model.last_wt, model.last_kappa,
             model.bivariate_normal_params],
            feed_dict)

        # sample mixture params
        s = sample_stroke(pi, e, mixture_bias, bv_params, std_bias)
        strokes.append(s)

    # append a pen lift at the end so that plotting works
    strokes[-1] = np.array([[[1, 0, 0]]])
    strokes = np.array(strokes).squeeze()
    return strokes


def _print_loss(model, data, epoch, stroke_length, char_length,
                batch_size, description=''):
    epoch_size = len(data[0]) // batch_size
    losses = []
    data_gen = conditional_batch_generator(
        data, stroke_length=stroke_length,
        char_length=char_length, batch_size=batch_size)
    for i in range(epoch_size):
        input_batch, target_batch, char_batch = next(data_gen)
        loss, mse = model.sess.run(
            [model.loss, model.mse],
            {model.inputs: input_batch, model.targets: target_batch,
             model.input_characters: char_batch})
        losses.append(loss)

    print('Epoch: {}, {} loss: {}'.format(
        epoch, description, np.mean(losses)))


@click.command()
@click.argument('model-folder')
@click.option('--num-epochs', default=500)
@click.option('--learning-rate', default=1e-4)
@click.option('--batch-size', default=32)
@click.option('--stroke-length', default=200)
@click.option('--steps-per-char', default=22,
              help='the avg number of stroke steps per character')
@click.option('--save-every', default=20)
def train(model_folder, num_epochs, learning_rate, batch_size,
          stroke_length, steps_per_char, save_every):

    char_seq_length = stroke_length // steps_per_char

    train_data, val_data, test_data, metadata = get_preprocessed_data_splits()
    train_data_gen = conditional_batch_generator(
        train_data, stroke_length=stroke_length,
        char_length=char_seq_length, batch_size=batch_size)

    model = ConditionalStrokeModel(
        model_folder, learning_rate=learning_rate,
        batch_size=batch_size, rnn_steps=stroke_length,
        is_train=True, char_dict=metadata['char_dict'],
        char_seq_len=char_seq_length)
    epoch_size = len(train_data[0]) // batch_size

    for epoch in range(num_epochs):
        for i in range(epoch_size):
            input_batch, target_batch, char_batch = next(train_data_gen)
            loss, _ = model.train(input_batch, target_batch, char_batch)

        if not epoch % save_every and epoch != 0:
            model.save()

            import os
            tf.reset_default_graph()
            model = ConditionalStrokeModel.load(
                model_folder, batch_size=1,
                rnn_steps=1, is_train=False,
                char_seq_len=30)
            strokes = decode(model, char_seq_len=30)

            plotting.plot_stroke(
                strokes, os.path.join(model_folder, 'test{}'.format(epoch)))
            tf.reset_default_graph()
            model = ConditionalStrokeModel.load(
                model_folder)

        _print_loss(model, train_data, epoch,
                    stroke_length, char_seq_length, batch_size, 'Train')
        _print_loss(model, val_data, epoch,
                    stroke_length, char_seq_length, batch_size, '    Val')
    model.save()


if __name__ == '__main__':
    train()
