import json
import os
import shutil
import traceback
from argparse import ArgumentParser

import yaml
from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL
import numpy as np

import models.simple_net
import train
from scripts.dataset import preprocess_dataset, analyze_dataset, visualize_analysis_results
from scripts.dataset.analyze_dataset import LABELS

script_dir = os.path.dirname(os.path.abspath(__file__))

workspace = ""
data_dir = ""
preprocess_config = {}
training_config = {}
analysis_results = {}


def objective(params):
    try:
        bounding_box_fbeta = params["bounding_box_fbeta"]
        ir_coverage = params["ir_coverage"]
        crop = params["crop"]
        dropout = params["dropout"]
        global workspace, data_dir, preprocess_config, training_config, analysis_results
        # preprocess data
        pp_data_dir = os.path.join(workspace, "data")
        if not os.path.exists(pp_data_dir):
            os.mkdir(pp_data_dir)
        f_scores = visualize_analysis_results.f_score(analysis_results["precisions"],
                                                      analysis_results["recalls"],
                                                      bounding_box_fbeta)
        i = np.argmax(f_scores)
        bounding_box = np.unravel_index(i, f_scores.shape) + np.array(analysis_results["bb_range"][0])
        if tuple(bounding_box) != tuple(preprocess_config["cropping"]["bb_size"]):
            preprocess_config["cropping"]["bb_size"] = bounding_box
            preprocess_dataset.main(preprocess_config, data_dir, pp_data_dir, do_resample=False, do_normalize=False)
        histogram = analyze_dataset.analyze(pp_data_dir, do_bounding_boxes=False)["histograms"][LABELS.index(crop)]
        _, i_min, i_max = visualize_analysis_results.crop_histogram(histogram, ir_coverage)
        if (i_min, i_max) != preprocess_config["normalization"][f"ir_{crop}"]:
            preprocess_config["normalization"][f"ir_{crop}"] = (i_min, i_max)
            preprocess_dataset.main(preprocess_config, data_dir, pp_data_dir, do_resample=False, do_crop=False)

        # train
        if os.path.exists(training_config["workspace"]):
            shutil.rmtree(training_config["workspace"])
            os.mkdir(training_config["workspace"])

        def get_model(input_shape):
            return models.simple_net.get_model(input_shape, dropout=dropout)
        train.main(training_config, custom_model_generator=get_model)
        with open(os.path.join(training_config["workspace"], "summary.json"), "r") as file:
            summary = json.load(file)
        return {
            "loss": -summary["auc_mean"],
            "status": STATUS_OK,
            "epochs": summary["epochs_mean"],
        }
    except Exception as e:
        return {
            "status": STATUS_FAIL,
            "attachments": {
                "exception": str(e)
            }
        }


def main(config):
    global workspace, data_dir, preprocess_config, training_config, analysis_results
    workspace = config["workspace"]
    data_dir = config["data"]
    with open(os.path.join(script_dir, "dataset", "preprocess.config")) as file:
        preprocess_config = yaml.load(file)
    with open(config["train_config"]) as file:
        training_config = yaml.load(file)
    pp_data_dir = os.path.join(workspace, "data")
    training_config["model"] = "custom"
    training_config["workspace"] = os.path.join(config["workspace"], "training")
    training_config["data"]["path"] = pp_data_dir
    if not os.path.exists(pp_data_dir):
        preprocess_dataset.main(preprocess_config, data_dir, pp_data_dir)
    analysis_results = analyze_dataset.analyze(pp_data_dir, do_histograms=False)
    analysis_results["precisions"] = np.array(analysis_results["precisions"])
    analysis_results["recalls"] = np.array(analysis_results["recalls"])
    f_beta_min, f_beta_max = config["search_space"]["bounding_box_fbeta"]
    ir_coverage_min, ir_coverage_max = config["search_space"]["ir_coverage"]
    dropout_min, dropout_max = config["search_space"]["dropout"]
    space = {"bounding_box_fbeta": hp.uniform("bounding_box_fbeta", f_beta_min, f_beta_max),
             "ir_coverage": hp.uniform("ir_coverage", ir_coverage_min, ir_coverage_max),
             "crop": hp.choice("crop", ["fixed", "roi", "seg"]),
             "dropout": hp.uniform("dropout", dropout_min, dropout_max)}
    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=config["max_evals"],
                trials_save_file=os.path.join(workspace, "save"))
    with open(os.path.join(workspace, "result.txt"), "w") as file:
        file.write(str(best))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", type=str, required=True)
    args = arg_parser.parse_args()
    with open(args.config, "r") as file:
        config_dict = yaml.load(file)
    main(config_dict)
