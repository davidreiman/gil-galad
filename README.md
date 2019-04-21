<p align="center">
  <img src="docs/images/logo.png">
</p>

---

<p align="center">
A deep learning project template for efficient, reproducible experiments in TensorFlow with Sacred.
</p>

```python

from sacred import Experiment
from sacred.observers import MongoObserver

from model import ResNet
from trainer import Trainer
from utils import TFRecordSampler, n_params


URL = '127.0.0.1'
DB_NAME = 'gil-galad'
USERNAME = 'username'
PASSWORD = 'password'
AUTHSOURCE = 'gil-galad'


"""Define an experiment and a MongoDB observer."""
ex = Experiment('gil-galad')
ex.observers.append(
    MongoObserver.create(
        url=URL,
        db_name=DB_NAME,
        username=USERNAME,
        password=PASSWORD,
        authSource=AUTHSOURCE,
    )
)


"""Define the experiment configuration."""
@ex.config
def config():
    learning_rate = 0.001
    n_batches = 1000
    batch_size = 32
    n_blocks = 3
    kernel_size = 3
    residual_filters = 32
    train_dir = './data/train'
    valid_dir = './data/valid'
    test_dir = './data/test'
    data_shapes = {
        'x': (32, 32, 3),
        'y': (128, 128, 3),
    }


"""Or a specific named configuration."""
@ex.named_config
def small():
    n_blocks = 1
    residual_filters = 16


"""The main function to execute an experiment."""
@ex.automain
def main(learning_rate, n_batches, batch_size, n_blocks, kernel_size,
    residual_filters, train_dir, valid_dir, test_dir, data_shapes,
    _run, _log):
    
    """Sacred seeds NumPy and TensorFlow automatically and delivers the seed to the MongoDB."""

    _log.info("Assembling graph...")

    sampler = TFRecordSampler(
        train_path=train_dir,
        valid_path=valid_dir,
        test_path=test_dir,
        data_shapes=data_shapes,
        batch_size=batch_size,
    )

    network = ResNet(
        kernel_size=kernel_size,
        residual_filters=residual_filters,
        n_blocks=n_blocks,
    )

    trainer = Trainer(
        network=network,
        sampler=sampler,
        learning_rate=learning_rate,
        run=_run,
        log=_log,
    )

    _log.info("Graph assembled. {} trainable parameters".format(n_params()))
    
    """The train method automatically sends TensorFlow logs and the latest checkpoint to Sacred."""
    
    trainer.train(
        n_batches=n_batches,
        summary_interval=5,
        checkpoint_interval=100,
    )

```
