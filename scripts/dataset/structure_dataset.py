import os
import re
import shutil
from argparse import ArgumentParser
from glob import glob
from shutil import copy2

patient_id_pattern = "UKD_[0-9]{4}$"
data_file_pattern = "{root}/CT/NRRD/CT_{id}.nrrd"
segmentation_file_pattern = "{root}/RoI/combined.nrrd"


def main(data, out, action, override):
    count = 0
    invalid_data_directories = []
    for root, directories, files in os.walk(data):
        try:
            patient_id = int(os.path.basename(root))
            if not re.match(patient_id_pattern, str(patient_id)):
                continue
        except ValueError:
            continue
        out_path = os.path.join(out, str(patient_id))
        if action != "none":
            if os.path.exists(out_path):
                if not override:
                    print(f"Skipping {out_path} (already exists)")
                    continue
                else:
                    shutil.rmtree(out_path)
            os.mkdir(out_path)
        data_file = glob(data_file_pattern.format(root=root, id=patient_id))
        segmentation_file = glob(segmentation_file_pattern.format(root=root, id=patient_id))
        if len(segmentation_file) != 1 or len(data_file) != 1:
            print(f"Invalid data directory: {root}. Ignoring")
            invalid_data_directories.append(root)
            if action != "none":
                shutil.rmtree(out_path)
            continue
        if action == "copy":
            copy2(data_file[0], os.path.join(out_path, "data.nrrd"))
            copy2(segmentation_file[0], os.path.join(out_path, "data.seg.nrrd"))
        elif action == "sym":
            os.symlink(data_file[0], os.path.join(out_path, "data.nrrd"))
            os.symlink(segmentation_file[0], os.path.join(out_path, "data.seg.nrrd"))
        count += 1
        print(f"{count} - Processed {root}")
    print("\n Invalid data directories:")
    print("\n".join(invalid_data_directories))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--data", "-d", type=str)
    arg_parser.add_argument("--out", "-o", type=str)
    arg_parser.add_argument("--action", "-a", type=str, choices=["copy", "sym", "none"], default="copy")
    arg_parser.add_argument("--override", "-ov", action="store_true")
    args = arg_parser.parse_args()
    main(args.data, args.out, args.action, args.override)

