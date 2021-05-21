import json
import os
from argparse import ArgumentParser

import tensorflow as tf
import yaml
from batchgenerators.transforms import RandomShiftTransform, MirrorTransform
from sklearn.model_selection import StratifiedKFold

from models import simple_net, squeeze_net
from data_loader import scan_data_directory, get_tf_dataset, get_data_augmenter

script_dir = os.path.dirname(os.path.abspath(__file__))
model_mapping = {
    "simplenet": simple_net.get_model,
    "squeezenet": squeeze_net.get_model,
    "custom": None,
}


def load_model(model_string, input_shape):
    return model_mapping[model_string](input_shape)


def get_transforms():
    return [MirrorTransform(),
            RandomShiftTransform(shift_mu=0, shift_sigma=3, p_per_channel=1)]


def _fit_batch_size(data_count, max_batch_size):
    for i in range(max_batch_size, 0, -1):
        if data_count % i == 0:
            return i


def main(config, custom_model_generator=None):
    model_mapping["custom"] = custom_model_generator
    config_training = config["training"]
    config_data = config["data"]
    dataset = scan_data_directory(config_data["path"], crop=config_data["crop"])
    k_fold = StratifiedKFold(n_splits=config_training["folds"], shuffle=False)
    sample_size = config_data["sample_size"] if config_data["sample"] else None
    if config_data["sample"]:
        input_shape = [*sample_size, 1]
    else:
        dummy_loader = get_data_augmenter(dataset[:1])
        input_shape = next(dummy_loader)[0].shape[-4:]
        dummy_loader._finish()
    transforms = get_transforms()
    # cross validation
    for i, (train, validation) in enumerate(k_fold.split(dataset, [x[0] for x in dataset]), start=1):
        checkpoint_dir = os.path.join(config["workspace"], "cross_validation", str(i))
        log_dir = os.path.join(checkpoint_dir, "logs")
        fold_summary_file = os.path.join(log_dir, "summary.json")
        if os.path.exists(fold_summary_file):
            continue
        # initialize datasets
        train = [dataset[i] for i in train]
        validation = [dataset[i] for i in validation]
        batch_size = _fit_batch_size(len(train), config_data["batch_size"])
        train_augmenter = get_data_augmenter(train, batch_size, sample_size, transforms, config_data["loader_threads"])
        train_dl = get_tf_dataset(train_augmenter, input_shape)
        #   cache validation data
        np.random.seed(seed=42)
        val_augmenter = get_data_augmenter(validation, len(validation), sample_size=sample_size)
        np.random.seed()
        validation_data = val_augmenter.__next__()
        val_augmenter._finish()
        # initialize model
        model = load_model(config["model"], input_shape)
        model.compile(optimizer=config_training["optimizer"],
                      loss=tf.losses.BinaryCrossentropy(from_logits=True),
                      metrics=["AUC"])

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="val_auc",
                                                                   patience=config_training["es_patience"]),
        callbacks = [early_stopping_callback]
        checkpoint_file = os.path.join(checkpoint_dir, "{epoch:04d}.ckpt")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                                 monitor="val_auc",
                                                                 save_weights_only=True,
                                                                 save_best_only=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)
        callbacks += [checkpoint_callback, tensorboard_callback]
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        initial_epoch = 0
        if latest_checkpoint is not None:
            model.load_weights(latest_checkpoint)
            initial_epoch = int(os.path.splitext(os.path.basename(latest_checkpoint))[0])
        # train
        tf.keras.backend.clear_session()
        history = model.fit(x=train_dl,
                            initial_epoch=initial_epoch,
                            epochs=config_training["epochs"],
                            validation_data=validation_data,
                            callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback])
        del model
        del train_dl
        del train_augmenter
        # save summary of fold
        fold_summary = {
            "auc": max(history.history["val_auc"]),
            "epochs": early_stopping_callback[0].stopped_epoch
        }
        with open(fold_summary_file, "w") as file:
            json.dump(fold_summary, file, indent=4)

    # save summary of cross validation
    cv_summary = {"auc_mean": 0, "epochs_mean": 0}
    for i in range(1, config_training["folds"] + 1):
        fold_summary_file = os.path.join(config["workspace"], "cross_validation", str(i), "logs", "summary.json")
        with open(fold_summary_file, "r") as file:
            fold_summary = json.load(file)
            cv_summary["auc_mean"] += fold_summary["auc"]
            cv_summary["epochs_mean"] += fold_summary["epochs"]
    cv_summary["auc_mean"] /= config_training["folds"]
    cv_summary["epochs_mean"] /= config_training["folds"]
    cv_summary_file = os.path.join(config["workspace"], "summary.json")
    with open(cv_summary_file, "w") as file:
        json.dump(cv_summary, file)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", default=os.path.join(script_dir, "default.config"))
    args = arg_parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    main(config)