import json
import os
from argparse import ArgumentParser

import tensorflow as tf
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from models import simple_net, squeeze_net, lombardo
from data_loader import get_dataset_from_config, get_data_loader_from_config

script_dir = os.path.dirname(os.path.abspath(__file__))
model_mapping = {
    "simplenet": simple_net.get_model,
    "lombardo": lombardo.get_model,
    "squeezenet": squeeze_net.get_model,
    "custom": None
}


def load_model(model_string, input_shape):
    if isinstance(input_shape, list) or isinstance(input_shape, tuple):
        if len(input_shape) == 1:
            ct_shape = input_shape[0]
            vector_shape = None
        else:
            ct_shape, vector_shape = input_shape
    else:
        ct_shape = input_shape
        vector_shape = None
    return model_mapping[model_string](ct_shape=ct_shape, vector_shape=vector_shape)


def train_model(config,
                train_data,
                validation_data,
                ct_shape,
                checkpoint_dir=None,
                log_dir=None):
    # initialize data loader
    config_training = config["training"]
    config_data = config["data"]
    train_dl, val_dl, input_shape = get_data_loader_from_config(train_data, validation_data, config_data, ct_shape)
    # initialize model
    model = load_model(config["model"], input_shape)
    model.compile(optimizer=config_training["optimizer"],
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=["AUC"])
    callbacks = []
    initial_epoch = 0
    if checkpoint_dir is not None:
        val_dl = val_dl.cache(os.path.join(checkpoint_dir, "val_cache"))
        checkpoint_file = os.path.join(checkpoint_dir, "{epoch:04d}.ckpt")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                                 monitor="val_auc",
                                                                 save_weights_only=True,
                                                                 save_best_only=True)
        callbacks.append(checkpoint_callback)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is not None:
            model.load_weights(latest_checkpoint)
            initial_epoch = int(os.path.splitext(os.path.basename(latest_checkpoint))[0])
    if log_dir is not None:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)
        callbacks.append(tensorboard_callback)
    # train
    tf.keras.backend.clear_session()
    return model.fit(x=train_dl,
                     initial_epoch=initial_epoch,
                     epochs=config_training["epochs"],
                     validation_data=val_dl,
                     callbacks=callbacks)


def get_history_from_tb(directory):
    history = {}
    for root, _, files in os.walk(directory):
        key = root[len(directory) + 1:]
        for file in files:
            if file[:6] == "events":
                file = os.path.join(root, file)
                for event in tf.compat.v1.train.summary_iterator(file):
                    for value in event.summary.value:
                        if value.tag not in history:
                            history[value.tag] = {}
                        if key not in history[value.tag]:
                            history[value.tag][key] = []
                        if not any(x[0] == event.step for x in history[value.tag][key]):
                            history[value.tag][key].append((event.step, value.simple_value))
    for tag in history.values():
        for series in tag:
            tag[series].sort(key=lambda x: x[0])
            tag[series] = [x[1] for x in tag[series]]
    return history


def main(config, custom_model_generator=None):
    model_mapping["custom"] = custom_model_generator
    config_training = config["training"]
    config_data = config["data"]
    dataset, ct_shape = get_dataset_from_config(config_data)

    # cross validation
    if config_training["folds"] >= 2:
        k_fold = StratifiedKFold(n_splits=config_training["folds"], shuffle=False)
        folds_done = 0
        history = get_history_from_tb(config["workspace"])
        if "epoch_loss" in history:
            history = history["epoch_loss"]
            for key in history:
                path = os.path.normpath(key).split(os.sep)
                if path[-1] == "train":
                    continue
                fold = int(path[1])
                if len(history[key]) == config_training["epochs"]:
                    folds_done += 1
                else:
                    folds_done = fold - 1
                    break

        for i, (train, validation) in enumerate(k_fold.split(dataset, [x[0] for x in dataset]), start=1):
            if i <= folds_done:
                continue
            train = [dataset[i] for i in train.tolist()]
            validation = [dataset[i] for i in validation.tolist()]
            checkpoint_dir = os.path.join(config["workspace"], "cross_validation", str(i))
            log_dir = os.path.join(checkpoint_dir, "logs")
            train_model(config, train, validation, ct_shape,
                        checkpoint_dir=checkpoint_dir,
                        log_dir=log_dir)

        # save summary of cross validation
        metrics = {
            "loss": [],
            "val_loss": [],
            "auc": [],
            "val_auc": []
        }
        tb_mapping = {
            "epoch_loss": {"train": "loss", "validation": "val_loss"},
            "epoch_auc": {"train": "auc", "validation": "val_auc"}
        }
        history = get_history_from_tb(config["workspace"])
        for tb_metric in history:
            if tb_metric in tb_mapping:
                for key in history[tb_metric]:
                    path = os.path.normpath(key).split(os.sep)
                    metrics[tb_mapping[tb_metric][path[-1]]].append(history[tb_metric][key])
        for m in metrics:
            metrics[m] = [sum(i) / config_training["folds"] for i in zip(*metrics[m])]
        with tf.summary.create_file_writer(os.path.join(config["workspace"], "train")).as_default():
            for i in range(config_training["epochs"]):
                tf.summary.scalar(f"avg_loss", metrics["loss"][i], step=i)
                tf.summary.scalar(f"avg_auc", metrics["auc"][i], step=i)
        with tf.summary.create_file_writer(os.path.join(config["workspace"], "val")).as_default():
            for i in range(config_training["epochs"]):
                tf.summary.scalar(f"avg_loss", metrics["val_loss"][i], step=i)
                tf.summary.scalar(f"avg_auc", metrics["val_auc"][i], step=i)
        with open(os.path.join(config["workspace"], "summary.json"), "w") as file:
            json.dump(
                {
                    "auc": max(metrics["auc"]),
                    "val_auc": max(metrics["val_auc"]),
                    "val_auc_epoch": metrics["val_auc"].index(max(metrics["val_auc"]) + 1)
                }, file)
    else:
        train, validation = train_test_split(dataset,
                                             train_size=config_training["train_size"],
                                             random_state=42,
                                             stratify=[x[0] for x in dataset])
        log_dir = os.path.join(config["workspace"], "logs")
        history = train_model(config, train, validation, ct_shape, checkpoint_dir=config["workspace"], log_dir=log_dir)
        fold_summary = {
            "loss": list(history.history["loss"]),
            "val_loss": list(history.history["val_loss"]),
            "auc": list(history.history["auc"]),
            "val_auc": list(history.history["val_auc"]),
        }
        with open(os.path.join(config["workspace"], "summary.json"), "w") as file:
            json.dump(fold_summary, file, indent=4)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", default=os.path.join(script_dir, "default.config"))
    args = arg_parser.parse_args()
    with open(args.config, "r") as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
    try:
        main(config_dict)
    finally:
        if "shutdown_system" in config_dict and config_dict["shutdown_system"]:
            os.system("shutdown now")
