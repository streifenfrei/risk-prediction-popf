import pickle
from argparse import ArgumentParser

import hyperopt

LABELS = ["full_bb", "roi", "seg"]


def main(trials: hyperopt.Trials):
    trials = [{"auc": -trial["result"]["loss"],
               "bb_beta": trial["misc"]["vals"]["bounding_box_fbeta"][0],
               "crop": LABELS[trial["misc"]["vals"]["crop"][0]],
               "dropout": trial["misc"]["vals"]["dropout"][0],
               "ci": trial["misc"]["vals"]["ir_coverage"][0],
               "avg_epochs_trained": trial["result"]["epochs"]
               }
              for trial in trials if trial["result"]["status"] == "ok"]
    trials.sort(key=lambda x: x["auc"], reverse=True)
    for trial in trials:
        print("AUC: {:.4f}, bb_beta: {:.2f}, crop: {}, dropout: {:.2f}, ci: {:.2f}, average epochs trained: {:.2f}".
              format(trial["auc"], trial["bb_beta"], trial["crop"], trial["dropout"], trial["ci"], trial["avg_epochs_trained"]))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--file", "-f", type=str, required=True)
    args = arg_parser.parse_args()
    with open(args.file, "rb") as file:
        trials = pickle.load(file)
    main(trials)
