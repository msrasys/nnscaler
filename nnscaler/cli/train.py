from .trainer import Trainer


if __name__ == '__main__':
    trainer = Trainer()
    if trainer.train_args.run_mode == 'run':
        trainer.train()

