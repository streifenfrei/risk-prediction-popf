import os
from abc import ABC

import SimpleITK as sitk
import numpy as np

from batchgenerators.dataloading import SlimDataLoaderBase, MultiThreadedAugmenter
from batchgenerators.transforms import AbstractTransform, Compose


def scan_data_directory(data_directory, crop="none"):
    assert crop in ["none", "fixed", "roi", "seg"]
    data = []
    for directory in os.scandir(data_directory):
        if directory.is_dir():
            try:
                patient_id = int(directory.name)
            except ValueError:
                continue
            # TODO load label
            import random
            label = random.choice([0, 1])

            data_full = os.path.join(directory.path, "full.nrrd")
            data_fixed = os.path.join(directory.path, "fixed.nrrd")
            data_roi = os.path.join(directory.path, "roi.nrrd")
            data_seg = os.path.join(directory.path, "seg.nrrd")
            segmentation = os.path.join(directory.path, "raw", "segmentation.seg.nrrd")
            if crop == "none" and os.path.exists(data_full):
                data.append((label, data_full, segmentation))
            elif crop == "fixed" and os.path.exists(data_fixed):
                data.append((label, data_fixed, segmentation))
            elif crop == "roi" and os.path.exists(data_roi):
                data.append((label, data_roi, segmentation))
            elif crop == "seg" and os.path.exists(data_seg):
                data.append((label, data_seg, segmentation))
    return data


class DataLoader(SlimDataLoaderBase, ABC):
    def __init__(self,
                 data,
                 batch_size,
                 number_of_threads_in_multithreaded=4,
                 epochs=1):
        super().__init__(None, batch_size, number_of_threads_in_multithreaded)
        self._data = data
        self._current_position = 0
        self.was_initialized = False
        self.epochs = epochs
        self._current_epoch = 0

    def initialize(self):
        self._current_epoch = 0
        self._reset()
        self.was_initialized = True

    def _reset(self):
        self._current_position = self.thread_id * self.batch_size

    def generate_train_batch(self):
        if not self.was_initialized:
            self.initialize()
        if self._current_position >= len(self._data):
            self._reset()
            self._current_epoch += 1
            if 0 < self.epochs <= self._current_epoch:
                raise StopIteration
        data_batch = []
        label_batch = []
        segmentation_batch = []
        for i in range(self.batch_size):
            index = self._current_position + i
            if index < len(self._data):
                label, data_file, segmentation_file = self._data[index]
                data_sitk = sitk.ReadImage(data_file)
                data_np = np.expand_dims(sitk.GetArrayFromImage(data_sitk).transpose(), axis=0)
                data_batch.append(data_np)
                label_batch.append(label)
                segmentation_sitk = sitk.ReadImage(segmentation_file)
                segmentation = [data_sitk.TransformPhysicalPointToIndex(segmentation_sitk.GetOrigin()),
                                segmentation_sitk.GetSize()]
                segmentation_batch.append(segmentation)
        batch = {"data": np.stack(data_batch),
                 "label": np.stack(label_batch),
                 "segmentation": np.stack(segmentation_batch)}

        self._current_position += self.number_of_threads_in_multithreaded * self.batch_size
        return batch


def get_data_loader_tf(data, batch_size=1, transforms=None, epochs=1, threads=1):
    transforms = [] if transforms is None else transforms
    threads = min(int(np.ceil(len(data) / batch_size)), threads)
    loader = DataLoader(data=data, batch_size=batch_size,
                        epochs=epochs, number_of_threads_in_multithreaded=threads)
    transforms = transforms + [PrepareForTF()]
    return MultiThreadedAugmenter(loader, Compose(transforms), threads)


class SampleFromSegmentation(AbstractTransform, ABC):
    def __init__(self, size, coverage):
        self.size = size
        self.coverage = coverage

    def __call__(self, **data_dict):
        # TODO implement
        return data_dict


class PrepareForTF(AbstractTransform, ABC):
    def __call__(self, **data_dict):
        data = data_dict["data"]
        data = np.moveaxis(data, 1, -1)
        label = data_dict["label"]
        return data, label
