import json
import os
from argparse import ArgumentParser

import tensorflow as tf
import yaml
from sklearn.model_selection import StratifiedKFold

from models import simple_net, squeeze_net
from data_loader import get_dataset_from_config, get_data_loader_from_config

script_dir = os.path.dirname(os.path.abspath(__file__))
model_mapping = {
    "simplenet": simple_net.get_model,
    "squeezenet": squeeze_net.get_model,
    "custom": None,
}


def load_model(model_string, input_shape):
    return model_mapping[model_string](input_shape)


def _prepare_config(config):
    config_training = config["training"]
    config_data = config["data"]
    config_data["sample_size"] = config_data["sample_size"] if config_data["sample"] else None
    config_data["sample_count"] = config_data["sample_count"] \
        if config_data["sample"] and not config_data["resize_to_sample_size"] else 1
    if config_data["sample"]:
        if config_data["crop"] in ["full", "fixed", "roi"]:
            config_data["crop"] = "roi_only"
        elif config_data["crop"] == "seg":
            config_data["crop"] = "roi_only_masked"
    config_data["blacklist"] = config_data["blacklist"] if "blacklist" in config_data else None
    return config, config_training, config_data


def main(config, custom_model_generator=None):
    model_mapping["custom"] = custom_model_generator
    config, config_training, config_data = _prepare_config(config)
    dataset, input_shape = get_dataset_from_config(config_data)
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
        train_dl, val_dl = get_data_loader_from_config(train, validation, config_data, input_shape)
        # initialize model
        model = load_model(config["model"], input_shape)
        model.compile(optimizer=config_training["optimizer"],
                      loss=tf.losses.BinaryCrossentropy(from_logits=True),
                      metrics=["AUC"])
        checkpoint_file = os.path.join(checkpoint_dir, "{epoch:04d}.ckpt")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                                 monitor="val_auc",
                                                                 save_weights_only=True)
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
            "auc": list(history.history["val_auc"]),
        }
        with open(fold_summary_file, "w") as file:
            json.dump(fold_summary, file, indent=4)

    # save summary of cross validation
    aucs = []
    for i in range(1, config_training["folds"] + 1):
        fold_summary_file = os.path.join(config["workspace"], "cross_validation", str(i), "logs", "summary.json")
        with open(fold_summary_file, "r") as file:
            fold_summary = json.load(file)
            aucs.append(fold_summary["auc"])
    aucs = [sum(i) / len(i) for i in zip(*aucs)]
    best_auc = max(aucs)
    epoch = aucs.index(best_auc)
    cv_summary = {
        "auc": best_auc,
        "epoch": epoch
    }
    cv_summary_file = os.path.join(config["workspace"], "summary.json")
    with open(cv_summary_file, "w") as file:
        json.dump(cv_summary, file)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", default=os.path.join(script_dir, "default.config"))
    args = arg_parser.parse_args()
    with open(args.config, "r") as config_file:
        config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
    main(config_dict)
