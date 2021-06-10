import json
import os
from argparse import ArgumentParser

import tensorflow as tf
import yaml
from sklearn.model_selection import StratifiedKFold

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


def main(config, custom_model_generator=None):
    model_mapping["custom"] = custom_model_generator
    config_training = config["training"]
    config_data = config["data"]
    dataset, ct_shape = get_dataset_from_config(config_data)
    k_fold = StratifiedKFold(n_splits=config_training["folds"], shuffle=False)
    # cross validation
    for i, (train, validation) in enumerate(k_fold.split(dataset, [x[0] for x in dataset]), start=1):
        checkpoint_dir = os.path.join(config["workspace"], "cross_validation", str(i))
        log_dir = os.path.join(checkpoint_dir, "logs")
        fold_summary_file = os.path.join(log_dir, "summary.json")
        if os.path.exists(fold_summary_file):
            continue
        # initialize data loader
        train = [dataset[i] for i in train.tolist()]
        validation = [dataset[i] for i in validation.tolist()]
        train_dl, val_dl, input_shape = get_data_loader_from_config(train, validation, config_data, ct_shape)
        # initialize model
        model = load_model(config["model"], input_shape)
        model.compile(optimizer=config_training["optimizer"],
                      loss=tf.losses.BinaryCrossentropy(),
                      metrics=["AUC"])
        checkpoint_file = os.path.join(checkpoint_dir, "{epoch:04d}.ckpt")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                                 monitor="val_auc",
                                                                 save_weights_only=True,
                                                                 save_best_only=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)
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
                            validation_data=val_dl,
                            callbacks=[checkpoint_callback, tensorboard_callback])
        # save summary of fold
        fold_summary = {
            "loss": list(history.history["loss"]),
            "val_loss": list(history.history["val_loss"]),
            "auc": list(history.history["auc"]),
            "val_auc": list(history.history["val_auc"]),
        }
        with open(fold_summary_file, "w") as file:
            json.dump(fold_summary, file, indent=4)

    # save summary of cross validation
    metrics = {
        "loss": [],
        "val_loss": [],
        "auc": [],
        "val_auc": []
    }
    sizes = [0 for _ in range(config_training["epochs"])]
    for i in range(1, config_training["folds"] + 1):
        fold_summary_file = os.path.join(config["workspace"], "cross_validation", str(i), "logs", "summary.json")
        with open(fold_summary_file, "r") as file:
            fold_summary = json.load(file)
            missing = config_training["epochs"] - len(fold_summary["auc"])
            for i in range(config_training["epochs"]):
                sizes[i] += 1 if i >= missing else 0
            for _ in range(missing):
                for m in metrics:
                    fold_summary[m].insert(0, 0)
            for m in metrics:
                metrics[m].append(fold_summary[m])
    for m in metrics:
        metrics[m] = [sum(i) / s for i, s in zip(zip(*metrics[m]), sizes)]
    with tf.summary.create_file_writer(os.path.join(config["workspace"], "train")).as_default():
        for i in range(config_training["epochs"]):
            tf.summary.scalar(f"avg_loss", metrics["loss"][i], step=i)
            tf.summary.scalar(f"avg_auc", metrics["auc"][i], step=i)
    with tf.summary.create_file_writer(os.path.join(config["workspace"], "val")).as_default():
        for i in range(config_training["epochs"]):
            tf.summary.scalar(f"avg_loss", metrics["val_loss"][i], step=i)
            tf.summary.scalar(f"avg_auc", metrics["val_auc"][i], step=i)
    cv_summary = {}
    for m in metrics:
        cv_summary[m] = max(metrics[m])
    cv_summary["val_loss_epoch"] = metrics["val_loss"].index(max(metrics["val_loss"]))
    cv_summary_file = os.path.join(config["workspace"], "summary.json")
    with open(cv_summary_file, "w") as file:
        json.dump(cv_summary, file)


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
