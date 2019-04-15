import os
import shutil
import tempfile
import numpy as np
import tensorflow as tf
from tqdm import tqdm


from utils import *


class Trainer:

    def __init__(self, network, sampler, learning_rate, run, log):
        """
        Builds graph and defines loss functions & optimizers. Handles automatic
        logging of metrics to Sacred.

        Args:
            network(models.Model): neural network model.
            sampler(utils.DataSampler): data sampler object.
        """

        self.network = network
        self.data = sampler
        self.learning_rate = learning_rate
        self.run = run
        self.log = log
        self.tmpdir = tempfile.mkdtemp()

        self.data.initialize()

        self.x, self.y = self.data.get_batch()

        self.y_ = self.network(self.x)

        self.loss = tf.losses.mean_squared_error(self.y, self.y_)
        self.eval_metric = tf.losses.mean_squared_error(self.y, self.y_)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            self.update = opt.minimize(
                loss=self.loss,
                var_list=self.network.vars,
                global_step=self.global_step
            )

        self.saver = tf.train.Saver(max_to_keep=1)

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.config = tf.ConfigProto(gpu_options=gpu_options)

        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

        self.summary_writer = tf.summary.FileWriter(
            logdir=self.tmpdir,
            graph=self.sess.graph
        )

    @property
    def artifacts(self):
        files = [os.path.join(self.tmpdir, file)
                for file in os.listdir(self.tmpdir)]
        return files

    def _train_mode(self):
        self.sess.run(self.data.get_dataset('train'))
        self.network.training = True

    def _eval_mode(self):
        self.sess.run(self.data.get_dataset('valid'))
        self.network.training = False

    def _valid_loss(self):
        valid_loss = self.evaluate()
        self._train_mode()
        return valid_loss

    def save(self):
        self.saver.save(
            sess=self.sess,
            save_path=os.path.join(self.tmpdir, 'ckpt'),
            global_step=self.global_step
        )

    def restore(self, filepath):
        meta_graph = [os.path.join(filepath, file) for file
            in os.listdir(filepath) if file.endswith('.meta')]
        restorer = tf.train.import_meta_graph(meta_graph[0])
        latest_ckpt = tf.train.latest_checkpoint(filepath)
        restorer.restore(self.sess, latest_ckpt)

    def summarize(self):
        """Add any metrics that should be visualized in OmniBoard here."""
        train_loss, global_step = self.sess.run([self.loss, self.global_step])
        valid_loss = self._valid_loss()

        self.run.log_scalar('training_loss', train_loss, global_step)
        self.run.log_scalar('validation_loss', valid_loss, global_step)

    def train(self, n_batches, summary_interval=100, checkpoint_interval=10000,
        restore_from_ckpt=None):

        self._train_mode()

        if restore_from_ckpt is not None:
            self.restore(restore_from_ckpt)

        try:
            for batch in tqdm(range(n_batches), unit='batch'):
                self.sess.run(self.update)

                if batch % summary_interval == 0:
                    self.summarize()

                if batch % checkpoint_interval == 0 or batch + 1 == n_batches:
                    self.save()

        except KeyboardInterrupt:
            self.save()
            self.log.info("Model saved. Training stopped.")

        finally:
            """Send logs and checkpoints to Sacred database."""
            for file in self.artifacts:
                self.run.add_artifact(file)
            shutil.rmtree(self.tmpdir)

    def evaluate(self):
        self._eval_mode()

        scores = []
        while True:
            try:
                metric = self.sess.run(self.loss)
                scores.append(metric)
            except tf.errors.OutOfRangeError:
                break

        mean_score = np.mean(scores)
        return mean_score
