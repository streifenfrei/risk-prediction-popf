import json
import os
from argparse import ArgumentParser

import tensorflow as tf
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt

from data_loader import get_data_loader_from_config, get_dataset_from_config
from models.vol_up_interpol import UpSampling3D
from train import load_model, get_history_from_tb

# https://stackoverflow.com/a/65523597
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)


# https://www.statworx.com/de/blog/erklaerbbarkeit-von-deep-learning-modellen-mit-grad-cam/
def get_heatmap(model, data):
    heatmaps = []
    for i in range(2):
        with tf.GradientTape() as tape:
            label, features = model(data)
            probs = label[:, i]
        grads = tape.gradient(probs, features)[0]
        output = features[0]
        grads = grads * tf.cast(output > 0, tf.float32) * tf.cast(grads > 0, tf.float32)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.zeros(grads.shape[0:3], dtype=tf.float32)
        for i, w in enumerate(weights):
            cam += w * output[:, :, :, i]
        scale = int(data.shape[1] / cam.shape[0])
        cam = UpSampling3D(size=(scale, scale, scale), interpolation="linear")(tf.expand_dims(tf.expand_dims(cam, 0), -1))
        cam = np.maximum(tf.squeeze(cam).numpy(), 0)
        heatmaps.append((cam - cam.min()) / (cam.max() - cam.min() + 10e-7))
    return heatmaps, label


def visualize(model, data_loader, ids, output_dir=None, evaluate_only=False, dataset_type="test"):
    auc = tf.keras.metrics.AUC()
    auc_p = tf.keras.metrics.AUC()
    auc_n = tf.keras.metrics.AUC()
    accuracy = tf.keras.metrics.Accuracy()
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for data, id in zip(data_loader, ids):
        (data), [gt_label, segmentation] = data
        [positive_hm, negative_hm], predicted_label = get_heatmap(model, data)
        auc.update_state(gt_label, predicted_label)
        if gt_label[0][0] == 1:
            auc_p.update_state(gt_label, predicted_label)
        elif gt_label[0][0] == 0:
            auc_n.update_state(gt_label, predicted_label)
        else:
            raise ValueError()
        accuracy.update_state(gt_label, predicted_label)
        if not evaluate_only:
            data = np.moveaxis(tf.squeeze(data).numpy(), (0, 1), (1, 0))
            positive_hm = np.moveaxis(tf.squeeze(positive_hm).numpy(), (0, 1), (1, 0))
            negative_hm = np.moveaxis(tf.squeeze(negative_hm).numpy(), (0, 1), (1, 0))
            segmentation = np.moveaxis(tf.squeeze(segmentation).numpy(), (0, 1), (1, 0))
            assert data.shape == positive_hm.shape == segmentation.shape
            for i in range(data.shape[-1]):
                fig, axs = plt.subplots(3, 2, constrained_layout=True)
                axs[0][0].imshow(data[:, :, i], cmap="gray", vmin=0, vmax=1)
                axs[0][0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                axs[0][1].imshow(data[:, :, i], cmap="gray", vmin=0, vmax=1)
                axs[0][1].imshow(positive_hm[:, :, i], cmap="jet", vmin=0, vmax=1, alpha=0.4)
                axs[0][1].imshow(segmentation[:, :, i], cmap="Oranges", vmin=0, vmax=1, alpha=0.4)
                axs[0][1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                axs[0][1].set_title(f"positive_prob: {predicted_label.numpy()[0, 0]}")
                axs[1][0].imshow(data[:, :, i], cmap="gray", vmin=0, vmax=1)
                axs[1][0].imshow(negative_hm[:, :, i], cmap="jet", vmin=0, vmax=1, alpha=0.4)
                axs[1][0].imshow(segmentation[:, :, i], cmap="Oranges", vmin=0, vmax=1, alpha=0.4)
                axs[1][0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                axs[1][0].set_title(f"negative_prob: {predicted_label.numpy()[0, 1]}")
                axs[1][1].imshow(segmentation[:, :, i], cmap="OrRd", vmin=0, vmax=1)
                axs[1][1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                axs[2][0].text(0.15, 0.4, f"id: {id}\n"
                                          f"dataset: {dataset_type}\n"
                                          f"ground truth: {gt_label.numpy()[0, 0]}/{gt_label.numpy()[0, 1]}")
                axs[2][0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                axs[2][0].axis("off")
                if output_dir is not None:
                    plt.savefig(os.path.join(output_dir, f"{id}_{i}.png"))
                else:
                    plt.show()
                plt.close()
    return auc.result().numpy().item(), auc_p.result().numpy().item(), auc_n.result().numpy().item(), accuracy.result().numpy().item()


def get_checkpoint_for_fold(workspace, history, fold):
    # take model with best validation AUC
    best_epoch = np.argmax(history["epoch_auc"][os.path.join("cross_validation", str(fold), "logs", "validation")]) + 1
    best_epoch = str(best_epoch).zfill(4)
    return os.path.join(workspace, "cross_validation", str(fold), f"{best_epoch.zfill(4)}.ckpt")


def main(config, evaluate_only=False):
    config["data"]["batch_size"] = 1
    volumetric = config["3D"]
    output_dir = os.path.join(config["workspace"], "evaluation")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    history = get_history_from_tb(config["workspace"])
    k_fold = StratifiedKFold(n_splits=config["training"]["folds"], shuffle=False)
    dataset, ct_shape = get_dataset_from_config(config, volumetric)
    aucs = []
    aucs_p = []
    aucs_n = []
    accs = []
    for i, (train, test) in enumerate(k_fold.split(dataset, [x[0] for x in dataset]), start=1):
        print(f"\rEvaluate fold {i}/{config['training']['folds']}", end="")
        train, validation = train_test_split(train,
                                             train_size=config["training"]["train_size"],
                                             random_state=42,
                                             stratify=[dataset[x][0] for x in train])
        train = [dataset[i] for i in train.tolist()]
        train_ids = [int(os.path.split(x[1])[0][-4:]) for x in train]
        validation = [dataset[i] for i in validation.tolist()]
        val_ids = [int(os.path.split(x[1])[0][-4:]) for x in validation]
        test = [dataset[i] for i in test.tolist()]
        test_ids = [int(os.path.split(x[1])[0][-4:]) for x in test]
        _, [train_loader, val_loader, test_loader], input_shape = get_data_loader_from_config(None,
                                                                                              [train, validation, test],
                                                                                              config, ct_shape,
                                                                                              volumetric=volumetric,
                                                                                              include_segmentation=True)
        extra_options = config.get("model_extra_options", {})
        model = load_model(config["model"], volumetric, input_shape, extra_options)
        checkpoint = tf.train.Checkpoint(model)
        checkpoint_file = get_checkpoint_for_fold(config["workspace"], history, i)
        checkpoint.restore(checkpoint_file).expect_partial()
        if config["model"] in ["simplenet", "resnet"]:
            model = tf.keras.models.Model([model.inputs], [model.output, model.get_layer("features").output])
        visualize(model, train_loader, train_ids, os.path.join(output_dir, "train", str(i)), evaluate_only, "training")
        visualize(model, val_loader, val_ids, os.path.join(output_dir, "val", str(i)), evaluate_only, "validation")
        auc, auc_p, auc_n, accuracy = visualize(model, test_loader, test_ids, os.path.join(output_dir, "test", str(i)), evaluate_only, "test")
        aucs.append(auc)
        aucs_p.append(auc_p)
        aucs_n.append(auc_n)
        accs.append(accuracy)
    with open(os.path.join(output_dir, "test", "result.json"), "w") as file:
        json.dump({
            "aucs": aucs,
            "avg_auc": sum(aucs) / len(aucs),
            "aucs_p": aucs_p,
            "avg_auc_p": sum(aucs_p) / len(aucs_p),
            "aucs_n": aucs_n,
            "avg_auc_n": sum(aucs_n) / len(aucs_n),
            "accuracies": accs,
            "avg_accuracy": sum(accs) / len(accs)
        }, file, indent=4)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", type=str, required=True)
    arg_parser.add_argument("--evaluate", "-e", action="store_true")
    args = arg_parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    main(config, args.evaluate)
