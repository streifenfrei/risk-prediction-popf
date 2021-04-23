import os
from argparse import ArgumentParser

import yaml
from data_loader import DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--config", "-c", default=os.path.join(script_dir, "default.config"))
    args = arg_parser.parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)["data"]
    data_loader = DataLoader(config["path"],
                             config["batch_size"],
                             config["crop"],
                             number_of_threads_in_multithreaded=1)
    for batch in data_loader:
        print(batch["data"].shape)


if __name__ == '__main__':
    main()
