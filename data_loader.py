import os
from abc import ABC

import SimpleITK as sitk
import numpy as np

from batchgenerators.dataloading import SlimDataLoaderBase
from batchgenerators.transforms import AbstractTransform


def scan_data_directory(data_directory, crop="none"):
    assert crop in ["none", "fixed", "roi", "seg"]
    data = []
    for directory in os.scandir(data_directory):
        if directory.is_dir():
            try:
                patient_id = int(directory.name)
            except ValueError:
                continue
            data_full = os.path.join(directory.path, "full.nrrd")
            data_fixed = os.path.join(directory.path, "fixed.nrrd")
            data_roi = os.path.join(directory.path, "roi.nrrd")
            data_seg = os.path.join(directory.path, "seg.nrrd")
            if crop == "none" and os.path.exists(data_full):
                data.append((patient_id, data_full))
            elif crop == "fixed" and os.path.exists(data_fixed):
                data.append((patient_id, data_fixed))
            elif crop == "roi" and os.path.exists(data_roi):
                data.append((patient_id, data_roi))
            elif crop == "seg" and os.path.exists(data_seg):
                data.append((patient_id, data_seg))
    return data


class DataLoader(SlimDataLoaderBase, ABC):
    def __init__(self,
                 data,
                 batch_size,
                 number_of_threads_in_multithreaded=4):
        super().__init__(None, batch_size, number_of_threads_in_multithreaded)
        self._data = data
        self.current_position = 0
        self.was_initialized = False

    def reset(self):
        self.current_position = self.thread_id * self.batch_size
        self.was_initialized = True

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        if self.current_position < len(self._data):
            data_batch = []
            label_batch = []
            for i in range(self.batch_size):
                index = self.current_position + i
                if index < len(self._data):
                    patient_id, data_file = self._data[index]
                    data_np = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(data_file)).transpose(), axis=0)
                    data_batch.append(data_np)
                    # TODO get label
                    label_batch.append([1])
            batch = {"data": np.stack(data_batch),
                     "label": np.stack(label_batch)}

            self.current_position += self.number_of_threads_in_multithreaded * self.batch_size
            return batch
        else:
            self.reset()
            raise StopIteration


class PrepareForTF(AbstractTransform, ABC):
    def __call__(self, **data_dict):
        # TODO implement
        return data_dict
