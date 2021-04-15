import logging
import os
from abc import ABC

import SimpleITK as sitk
import numpy as np

from batchgenerators.dataloading import SlimDataLoaderBase

class DataLoader(SlimDataLoaderBase, ABC):
    def __init__(self, data_directory, batch_size, crop="none", number_of_threads_in_multithreaded=4):
        super().__init__(None, batch_size, number_of_threads_in_multithreaded)
        assert crop in ["none", "roi", "seg"]
        self.crop = crop
        self._data = []
        for root, _, files in os.walk(data_directory):
            try:
                patient_id = int(os.path.basename(root))
            except ValueError:
                continue
            nrrd_files = set([x for x in files if x[-4:] == "nrrd"])
            segmentation_file = set([x for x in nrrd_files if x[-8:] == "seg.nrrd"])
            data_file = nrrd_files - segmentation_file
            if len(segmentation_file) != 1 or len(data_file) != 1:
                logging.warning(f"Amount of .nrrd files found in {root} != 2. Skipping")
                continue
            data_file = os.path.join(root, list(data_file)[0])
            segmentation_file = os.path.join(root, list(segmentation_file)[0])
            self._data.append((patient_id, data_file, segmentation_file))
        self.current_position = 0
        self.was_initialized = False

    def reset(self):
        self.current_position = self.thread_id * self.batch_size
        self.was_initialized = True

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        if self.current_position < len(self._data):
            batch_sitk = []
            for i in range(self.batch_size):
                index = self.current_position + i
                if index < len(self._data):
                    patient_id, data_file, segmentation_file = self._data[self.current_position]
                    data_sitk = sitk.ReadImage(data_file, imageIO="NrrdImageIO")
                    segmentation_sitk = None
                    if self.crop != "none":
                        segmentation_sitk = sitk.ReadImage(data_file, imageIO="NrrdImageIO")
                    batch_sitk.append((patient_id, data_sitk, segmentation_sitk))

            batch_np = []
            if self.crop == "none":
                for patient_id, data_sitk, _ in batch_sitk:
                    data_np = sitk.GetArrayFromImage(data_sitk)
                    data_np = np.expand_dims(np.moveaxis(data_np, 0, 2), axis=0)
                    batch_np.append(data_np)

            batch = {"data": np.stack(batch_np)}

            self.current_position += self.number_of_threads_in_multithreaded * self.batch_size
            return batch
        else:
            self.reset()
            raise StopIteration
