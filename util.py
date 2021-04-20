import os


def data_iterator(data_directory):
    for root, _, files in os.walk(data_directory):
        try:
            patient_id = int(os.path.basename(root))
        except ValueError:
            continue
        nrrd_files = set([x for x in files if x[-4:] == "nrrd"])
        segmentation_file = set([x for x in nrrd_files if x[-8:] == "seg.nrrd"])
        data_file = nrrd_files - segmentation_file
        if len(segmentation_file) != 1 or len(data_file) != 1:
            print(f"Invalid data directory {root}. Ignoring")
            continue
        data_file = os.path.join(root, list(data_file)[0])
        segmentation_file = os.path.join(root, list(segmentation_file)[0])
        yield patient_id, data_file, segmentation_file