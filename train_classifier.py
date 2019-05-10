"""Script for training Source Separation Unet."""

from argparse import ArgumentParser

from trainers.classifier_trainer import ClassifierTrainer

if __name__ == '__main__':
    """
    Usage:
        python train_classifier.py \
            --model_type basic
            --data_path /scratch/zh1115/dsga1008/ssl_data_96 \
            --save_dir saved_models
    """

    ap = ArgumentParser()
    ap.add_argument("-mt", "--model_type", default='vasic',
                    help="Name of model to use.")
    ap.add_argument("-nc", "--n_classes", type=int, default=1000,
                    help="Number of classes to predict.")
    ap.add_argument("-bs", "--batch_size", type=int, default=8,
                    help="Batch size for optimization.")
    ap.add_argument("-lr", "--learning_rate", type=float, default=0.0001,
                    help="Learning rate for optimization.")
    ap.add_argument("-ne", "--num_epochs", type=int, default=100,
                    help="Number of epochs for optimization.")
    ap.add_argument("-pt", "--patience", type=int, default=10,
                    help="Number of to wait before reducing learning rate.")
    ap.add_argument("-ep", "--eval_pct", type=float, default=0.05,
                    help="amount of Val set to evaluate")
    ap.add_argument("-ml", "--min_lr", type=float, default=0.0,
                    help="Minimum learning rate.")
    ap.add_argument("-td", "--data_path",
                    help="Location of images.")
    ap.add_argument("-sd", "--save_dir",
                    help="Location to save the model.")
    ap.add_argument("-pw", "--pretrain_weights",
                    help="model file with the pretrained weights.")
    ap.add_argument("-wd", "--weight_decay", type=float, default=1e-6,
                    help="Weight decay for nonbert models.")
    ap.add_argument("-ch", "--classifier_hidden", type=int, default=2048,
                    help="Size of classifier hidden layer")
    ap.add_argument("-es", "--early_stop", type=int, default=5,
                    help="number of epochs of no improvement to stop at")
    ap.add_argument("-sc", "--simple_cls", action='store_true')

    args = vars(ap.parse_args())

    trainer = ClassifierTrainer(model_type=args['model_type'],
                     n_classes=args['n_classes'],
                     batch_size=args['batch_size'],
                     learning_rate=args['learning_rate'],
                     num_epochs=args['num_epochs'],
                     weight_decay=args['weight_decay'],
                     patience=args['patience'],
                     min_lr=args['min_lr'],
                     pretrain_weights=args['pretrain_weights'],
                     cls_hid_dim = args['classifier_hidden'],
                     eval_pct= args['eval_pct'],
                     early_stopping = args['early_stop'],
                     simple_cls = args['simple_cls']
                    )


    trainer.fit(args['data_path'], args['save_dir'])
