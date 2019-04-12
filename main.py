import sacred
import gilgalad as gg
from sacred.observers import FileStorageObserver as obs


ex = sacred.Experiment('gil-galad')
ex.observers.append(obs.create('/Users/David/Documents/Projects/sacred-workflow/runs'))


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
    logdir = '/Users/David/Documents/project/logdir'
    ckptdir = '/Users/David/Documents/project/ckptdir'


"""For reference: https://github.com/maartjeth/sacred-example-pytorch/blob/master/train_nn.py
Uses a similar structure but puts the "Trainer" class in the top-level module
which allows it to capture _run to log metrics... here, we capture _run and _log
in the main function and pass it to the graph class."""


@ex.automain
def main(learning_rate, n_batches, batch_size, n_blocks, kernel_size,
    residual_filters, train_dir, valid_dir, test_dir, logdir, ckptdir,
    _run, _log):

    data_shapes = {
        'x': (32, 32, 3),
        'y': (128, 128, 3),
    }

    _log.info("Assembling graph...")

    sampler = gg.utils.TFRecordSampler(
        train_path=train_dir,
        valid_path=valid_dir,
        test_path=test_dir,
        data_shapes=data_shapes,
        batch_size=batch_size,
    )

    network = gg.models.ResNet(
        kernel_size=3,
        residual_filters=32,
        n_blocks=3,
    )

    trainer = gg.trainer.Trainer(
        network=network,
        sampler=sampler,
        learning_rate=learning_rate,
        logdir=logdir,
        ckptdir=ckptdir,
        run=_run,
        log=_log,
    )

    _log.info("Graph assembled.")

    trainer.train(n_batches=n_batches)
