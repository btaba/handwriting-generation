"""
Recreate character generation even though dataset is super small.
Then we can use the same network
for handwritten generation with the added mixture density network
"""
import click
import numpy as np
import tensorflow as tf
from handwriting_gen.data_manager import (
    get_preprocessed_data_splits, character_batch_generator
)
from handwriting_gen.networks import stacked_lstm_model, get_embedding, TFModel


class CharacterGenModel(TFModel):

    def __init__(self, save_path, batch_size, rnn_steps, vocab_size,
                 num_layers, hidden_size, max_grad_norm=5, **kwargs):
        """
        """
        super().__init__(save_path)
        self.sess = tf.Session()

        self.batch_size = batch_size
        self.rnn_steps = rnn_steps
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.max_grad_norm = max_grad_norm

        self.build()
        self.sess.run(tf.global_variables_initializer())

    def build(self):

        self.inputs = inputs = tf.placeholder(
            tf.int32, [self.batch_size, self.rnn_steps], 'input')
        self.targets = tf.placeholder(
            tf.int32, [self.batch_size, self.rnn_steps], 'targets')

        with tf.variable_scope('character-gen', reuse=tf.AUTO_REUSE):
            embedding = get_embedding(
                inputs, self.vocab_size, self.hidden_size)

            embedding = [tf.squeeze(i, [1]) for i in
                         tf.split(embedding, self.rnn_steps, axis=1)]

            outputs, self.last_state, self.intial_state = stacked_lstm_model(
                embedding, self.hidden_size, self.num_layers,
                self.batch_size)

            output = tf.reshape(
                tf.concat(outputs[-1], 1), [-1, self.hidden_size])

            logits = tf.layers.dense(output, self.vocab_size)
            logits = tf.reshape(
                logits, [self.batch_size, self.rnn_steps, self.vocab_size])
            self.logits = logits

        self.build_loss()
        self.saver = tf.train.Saver()

    def build_loss(self):
        loss = tf.contrib.seq2seq.sequence_loss(
            self.logits,
            self.targets,
            tf.ones([self.batch_size, self.rnn_steps], dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)

        cost = tf.reduce_sum(loss)
        self.lr = lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables('character-gen')
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          self.max_grad_norm)
        self.optimizer = tf.train.GradientDescentOptimizer(lr)
        self.train_op = self.optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        self.new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_lr")
        self.lr_update = tf.assign(lr, self.new_lr)
        self.cost = cost

    def train(self, inputs, targets):
        return self.sess.run(
            [self.cost, self.train_op],
            {self.inputs: inputs, self.targets: targets})


def decode(model, metadata, steps=100, seed=42):
    d = {v: k for k, v in metadata['char_dict'].items()}

    chars = []
    feed_dict = {}
    last_state = model.sess.run(model.intial_state)
    x = np.array([[seed]])

    for _ in range(steps):
        feed_dict[model.inputs] = x
        for idx, i in enumerate(model.intial_state):
            feed_dict[i] = last_state[idx]

        out, last_state = model.sess.run(
            [model.logits, model.last_state], feed_dict)

        x = np.argmax(out)
        chars.append(d[x])
        x = [[x]]

    print(''.join(chars))


@click.command()
@click.argument('model-folder')
@click.option('--learning-rate', default=1)
@click.option('--num-epochs', default=50)
@click.option('--num-layers', default=2)
@click.option('--num-steps', default=30,
              help='number of time-steps in rollout')
@click.option('--hidden-size', default=200)
@click.option('--lr-decay', default=0.0001)
@click.option('--batch-size', default=32)
def train(model_folder, learning_rate, num_epochs,
          num_layers, num_steps, hidden_size,
          lr_decay, batch_size):

    train_data, _, test_data, metadata = get_preprocessed_data_splits(
        sentences_to_int=True)

    train_data_gen = character_batch_generator(
        train_data, batch_size=batch_size)

    vocab_size = len(metadata['char_dict'])
    epoch_size = len(train_data[0]) // batch_size

    model = CharacterGenModel(
        model_folder, batch_size, num_steps, vocab_size,
        num_layers, hidden_size)

    for i in range(num_epochs):
        learning_rate *= 1 / (1. + lr_decay * i)
        model.sess.run(model.lr_update, {model.new_lr: learning_rate})

        print('Epoch {}, lr {}'.format(i, model.sess.run(model.lr)))

        total_cost = 0
        for _ in range(epoch_size):
            input_batch, target_batch = next(train_data_gen)
            cost = model.train(input_batch, target_batch)[0]
            total_cost += cost

        model.save()
        train_perplexity = np.exp(total_cost / (epoch_size))
        print('Epoch: {} Train perplexity: {}'.format(i, train_perplexity))

    total_cost = 0
    test_epochs = len(test_data[0]) // batch_size
    for _ in range(test_epochs):
        test_input, test_target = next(character_batch_generator(test_data))
        cost = model.sess.run(
            model.cost,
            {model.inputs: test_input, model.targets: test_target})
        total_cost += cost

    test_perplexity = np.exp(total_cost / test_epochs)
    print("Test perplexity: {}".format(test_perplexity))
    print('BPC: {}'.format(total_cost / test_epochs))


if __name__ == '__main__':
    train()
