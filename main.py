import json
import os
import random
from argparse import ArgumentParser

import tensorflow as tf
import numpy as np
import yaml
from batchgenerators.transforms import RandomShiftTransform, MirrorTransform
from sklearn.model_selection import KFold

from models import simple_net, squeeze_net
from data_loader import scan_data_directory, SampleFromSegmentation, get_data_loader_tf

script_dir = os.path.dirname(os.path.abspath(__file__))
model_mapping = {
    "simplenet": simple_net.get_model,
    "squeezenet": squeeze_net.get_model
}


def load_model(model_string, input_shape):
    return model_mapping[model_string](input_shape)


def get_transforms():
    return [MirrorTransform(),
            RandomShiftTransform(shift_mu=0, shift_sigma=3, p_per_channel=1)]


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", default=os.path.join(script_dir, "default.config"))
    args = arg_parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config_training = config["training"]
    config_data = config["data"]
    if config_data["sample"]:
        config_data["crop"] = "none"
    full_dataset = scan_data_directory(config_data["path"], crop=config_data["crop"])
    dataset_true = [x for x in full_dataset if x[0] == 1]
    dataset_false = [x for x in full_dataset if x[0] == 0]
    k_fold = KFold(n_splits=config_training["folds"], shuffle=False)
    # augmentation
    transforms = []
    if config_data["sample"]:
        transforms.append(SampleFromSegmentation(config_data["sample_size"], config_data["sample_coverage"]))
        input_shape = [*config_data["sample_size"], 1]
    else:
        dummy_loader = get_data_loader_tf(full_dataset[:1])
        input_shape = next(dummy_loader)[0].shape[-4:]
        dummy_loader._finish()
    transforms += get_transforms()
    # cross validation
    for i, ((train_t, validation_t), (train_f, validation_f)) \
            in enumerate(zip(k_fold.split(dataset_true), k_fold.split(dataset_false)), start=1):
        checkpoint_dir = os.path.join(config["workspace"], "cross_validation", str(i))
        log_dir = os.path.join(checkpoint_dir, "logs")
        fold_summary_file = os.path.join(log_dir, "summary.json")
        if os.path.exists(fold_summary_file):
            continue
        # initialize datasets
        train = [dataset_true[x] for x in train_t] + [dataset_false[x] for x in train_f]
        validation = [dataset_true[x] for x in validation_t] + [dataset_false[x] for x in validation_f]
        if config_data["sample"]:
            train *= config_data["sample_count"]
            validation *= config_data["sample_count"]
        random.shuffle(train)
        random.shuffle(validation)
        dl_train = get_data_loader_tf(train, config_data["batch_size"], transforms,
                                      config_training["epochs"], config_data["loader_threads"])
        train_steps = np.ceil(len(train) / config_data["batch_size"])
        #   cache validation data
        dl_validation = get_data_loader_tf(validation, len(validation))
        validation_data = dl_validation.__next__()
        dl_validation._finish()
        # initialize model
        model = load_model(config["model"], input_shape)
        model.compile(optimizer=config_training["optimizer"],
                      loss=tf.losses.BinaryCrossentropy(from_logits=True),
                      metrics=["AUC"])
        checkpoint_file = os.path.join(checkpoint_dir, "{epoch:04d}.ckpt")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                                 monitor="val_auc",
                                                                 save_weights_only=True,
                                                                 save_best_only=True)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="val_auc",
                                                                   patience=config_training["es_patience"]),
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        initial_epoch = 0
        if latest_checkpoint is not None:
            model.load_weights(latest_checkpoint)
            initial_epoch = int(os.path.splitext(os.path.basename(latest_checkpoint))[0])
        # train
        tf.keras.backend.clear_session()
        history = model.fit(x=dl_train,
                            initial_epoch=initial_epoch,
                            epochs=config_training["epochs"],
                            validation_data=validation_data,
                            steps_per_epoch=train_steps,
                            callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback])
        dl_train._finish()
        # save summary of fold
        fold_summary = {
            "auc": max(history.history["val_auc"]),
            "epochs": early_stopping_callback[0].stopped_epoch
        }
        with open(fold_summary_file, "w") as file:
            json.dump(fold_summary, file, indent=4)

    # save summary of cross validation
    cv_summary = {"auc_mean": 0, "epochs_mean": 0}
    for i in range(1, config_training["folds"]+1):
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
    main()
