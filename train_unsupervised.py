"""Script for training Source Separation Unet."""

from argparse import ArgumentParser

from trainers.unsupervised_trainer import UnsupervisedTrainer

if __name__ == '__main__':
    """
    Usage:
        python train_classifier.py \
            --model_type basic
            --data_path /scratch/zh1115/dsga1008/ssl_data_96 \
            --save_dir saved_models
    """

    ap = ArgumentParser()
    ap.add_argument("-mt", "--model_type", default='gan',
                    help="Name of model to use.")
    ap.add_argument("-bs", "--batch_size", type=int, default=8,
                    help="Batch size for optimization.")
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.00001,
                    help="Learning rate for optimization.")
    ap.add_argument("-ne", "--num_epochs", type=int, default=100,
                    help="Number of epochs for optimization.")
    ap.add_argument("-pt", "--patience", type=int, default=10,
                    help="Number of to wait before reducing learning rate.")
    ap.add_argument("-ml", "--min_lr", type=float, default=0.0,
                    help="Minimum learning rate.")
    ap.add_argument("-td", "--data_path",
                    help="Location of images.")
    ap.add_argument("-sd", "--save_dir",
                    help="Location to save the model.")
    ap.add_argument("-wd", "--weight_decay", type=float, default=1e-6,
                    help="Weight decay for nonbert models.")

    args = vars(ap.parse_args())

    trainer = UnsupervisedTrainer(model_type=args['model_type'],
                     batch_size=args['batch_size'],
                     learning_rate=args['learning_rate'],
                     num_epochs=args['num_epochs'],
                     weight_decay=args['weight_decay'],
                     patience=args['patience'],
                     min_lr=args['min_lr'],
                            )

    trainer.fit(args['data_path'], args['save_dir'])
