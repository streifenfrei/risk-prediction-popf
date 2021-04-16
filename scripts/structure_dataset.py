import os
import re
import shutil
from argparse import ArgumentParser
from glob import glob
from shutil import copy2

patient_id_pattern = "[0-9]{4}$"


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--data", "-d", type=str)
    arg_parser.add_argument("--out", "-o", type=str)
    arg_parser.add_argument("--action", "-a", type=str, choices=["copy", "sym", "none"], default="copy")
    arg_parser.add_argument("--override", "-ov", action="store_true")
    args = arg_parser.parse_args()
    count = 0
    invalid_data_directories = []
    for root, directories, files in os.walk(args.data):
        try:
            patient_id = int(os.path.basename(root))
            if not re.match(patient_id_pattern, str(patient_id)):
                continue
        except ValueError:
            continue
        out_path = os.path.join(args.out, str(patient_id))
        if args.action != "none":
            if os.path.exists(out_path):
                if not args.override:
                    print(f"Skipping {out_path} (already exists)")
                    continue
                else:
                    shutil.rmtree(out_path)
            os.mkdir(out_path)
        if "CT" in directories and "ROI" in directories:
            data_file = glob(f"{root}/CT/*.nrrd")
            segmentation_file = glob(f"{root}/ROI/*.seg.nrrd")
        else:
            nrrd_files = set(glob(f"{root}/*.nrrd"))
            segmentation_file = set([x for x in nrrd_files if x[-8:] == "seg.nrrd"])
            data_file = list(nrrd_files - segmentation_file)
            segmentation_file = list(segmentation_file)
        if len(segmentation_file) != 1 or len(data_file) != 1:
            print(f"Invalid data directory: {root}. Ignoring")
            invalid_data_directories.append(root)
            if args.action != "none":
                shutil.rmtree(out_path)
            continue
        if args.action == "copy":
            copy2(data_file[0], os.path.join(out_path, "data.nrrd"))
            copy2(segmentation_file[0], os.path.join(out_path, "data.seg.nrrd"))
        elif args.action == "sym":
            os.symlink(data_file[0], os.path.join(out_path, "data.nrrd"))
            os.symlink(segmentation_file[0], os.path.join(out_path, "data.seg.nrrd"))
        count += 1
        print(f"{count} - Processed {root}")
    print("\n Invalid data directories:")
    print("\n".join(invalid_data_directories))


if __name__ == '__main__':
    main()
