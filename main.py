import sacred
from model import ResNet
from trainer import Trainer
from utils import TFRecordSampler, n_params, get_trainable_params


ex = sacred.Experiment('gil-galad')
observer = sacred.observers.FileStorageObserver.create('/Users/David/Documents/Projects/sacred-workflow/runs')
ex.observers.append(observer)


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


@ex.automain
def main(learning_rate, n_batches, batch_size, n_blocks, kernel_size,
    residual_filters, train_dir, valid_dir, test_dir, data_shapes,
    _run, _log):

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

    trainer.train(
        n_batches=n_batches,
        summary_interval=5,
        checkpoint_interval=100,
    )
