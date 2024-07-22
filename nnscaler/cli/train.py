import logging

import nnscaler

from .trainer import Trainer


if __name__ == '__main__':
    nnscaler.utils.set_default_logger_level(level=logging.INFO)
    trainer = Trainer()
    if trainer.train_args.run_mode == 'run':
        trainer.train()
