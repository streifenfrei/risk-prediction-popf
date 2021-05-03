import json
import os
from argparse import ArgumentParser

import tensorflow as tf
import yaml
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.transforms import Compose, RandomShiftTransform, MirrorTransform
from sklearn.model_selection import KFold

from models import simple_net, squeeze_net
from data_loader import DataLoader, scan_data_directory, PrepareForTF

script_dir = os.path.dirname(os.path.abspath(__file__))
model_mapping = {
    "simplenet": simple_net.get_model,
    "squeezenet": squeeze_net.get_model
}


def load_model(model_string, input_shape):
    return model_mapping[model_string](input_shape)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", default=os.path.join(script_dir, "default.config"))
    args = arg_parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config_training = config["training"]
    config_data = config["data"]

    full_dataset = scan_data_directory(config_data["path"], crop=config_data["crop"])
    k_fold = KFold(n_splits=config_training["folds"], shuffle=False)
    # get input shape from dummy loader
    dummy_loader = MultiThreadedAugmenter(DataLoader(data=full_dataset[:1], batch_size=1), PrepareForTF(), 1)
    input_shape = next(dummy_loader)[0].shape[-4:]
    # input_shape = [*[int(2**np.ceil(np.log2(x))) for x in input_shape[:-1]], 1]
    # transforms.append(ResizeTransform(input_shape[:-1]))
    transforms = [MirrorTransform(),
                  RandomShiftTransform(shift_mu=0, shift_sigma=3, p_per_channel=1),
                  PrepareForTF()]
    transforms = Compose(transforms)
    for i, (train, validation) in enumerate(k_fold.split(full_dataset)):
        checkpoint_dir = os.path.join(config["workspace"], "cross_validation", str(i))
        log_dir = os.path.join(checkpoint_dir, "logs")
        history_file = os.path.join(log_dir, "history.json")
        if os.path.exists(history_file):
            continue
        dl_train = MultiThreadedAugmenter(DataLoader(data=[full_dataset[x] for x in train],
                                                     batch_size=config_data["batch_size"],
                                                     number_of_threads_in_multithreaded=config_data["loader_threads"]),
                                          transforms,
                                          config_data["loader_threads"])
        dl_validation = MultiThreadedAugmenter(DataLoader(data=[full_dataset[x] for x in validation],
                                                          batch_size=config_data["batch_size"],
                                                          number_of_threads_in_multithreaded=config_data[
                                                              "loader_threads"]),
                                               transforms,
                                               config_data["loader_threads"])
        model = load_model(config["model"], input_shape)
        model.summary()
        model.compile(optimizer=config_training["optimizer"],
                      loss=tf.losses.BinaryCrossentropy(from_logits=True),
                      metrics=["auc"])
        checkpoint_file = os.path.join(checkpoint_dir, "{epoch:04d}.ckpt")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                                 save_weights_only=True,
                                                                 save_best_only=True)
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="auc",
                                                                   patience=config_training["es_patience"])
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        initial_epoch = 0
        if latest_checkpoint is not None:
            model.load_weights(latest_checkpoint)
            initial_epoch = int(os.path.splitext(os.path.basename(latest_checkpoint))[0])
        history = model.fit(x=dl_train,
                            initial_epoch=initial_epoch,
                            epochs=config_training["epochs"],
                            validation_data=dl_validation,
                            callbacks=[checkpoint_callback, early_stopping_callback, tensorboard_callback])
        with open(history_file, "w") as file:
            json.dump(history.history, file)


if __name__ == '__main__':
    main()
