import os
from argparse import ArgumentParser

import yaml
import matplotlib.pyplot as plt
import numpy as np
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
                             roi=[*config["roi_root"], *config["roi_size"]])
    for batch in data_loader:
        data = batch["data"]
        for image in np.split(data, indices_or_sections=data.shape[-1], axis=-1):
            image = image.squeeze()
            for img in np.split(image, indices_or_sections=image.shape[0], axis=0):
                plt.imshow(img.transpose())
                plt.show()


if __name__ == '__main__':
    main()
