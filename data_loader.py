import csv
import os
from abc import ABC

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop

from batchgenerators.dataloading import SlimDataLoaderBase, MultiThreadedAugmenter
from batchgenerators.transforms import AbstractTransform, Compose

from scripts.dataset.preprocess_dataset import Crop


def scan_data_directory(data_directory, crop="none"):
    assert crop in ["none", "fixed", "roi", "seg", "all"]
    labels = {}
    with open(os.path.join(data_directory, "labels.csv"), "r") as file:
        reader = csv.reader(file)
        for row in reader:
            labels[int(row[0])] = int(row[1])
    data = []
    for directory in os.scandir(data_directory):
        if directory.is_dir():
            try:
                patient_id = int(directory.name)
            except ValueError:
                continue
            if patient_id not in labels:
                print(f"No label found for patient {patient_id}")
                continue
            label = labels[patient_id]

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
            elif crop == "all":
                data.append(directory.path)
    return data


class DataLoader(SlimDataLoaderBase, ABC):
    def __init__(self,
                 data,
                 batch_size,
                 sample_size=None,
                 number_of_threads_in_multithreaded=4,
                 epochs=1):
        super().__init__(None, batch_size, number_of_threads_in_multithreaded)
        self._data = data
        self._current_position = 0
        self.sample_size = sample_size
        self.was_initialized = False
        self.epochs = epochs
        self._current_epoch = 0

    def initialize(self):
        self._current_epoch = 0
        self._reset()
        self.was_initialized = True

    def _reset(self):
        self._current_position = self.thread_id * self.batch_size

    def _sample(self, data: sitk.Image, segmentation: sitk.Image):
        roi_size = segmentation.GetSize()
        crop_size = [max(s_sz, roi_size) for s_sz, roi_sz in zip(self.sample_size, roi_size)]
        crop = sitk.GetArrayFromImage(Crop(data, segmentation, crop_size).fixed())
        crop = np.expand_dims(np.expand_dims(crop, 0), 0)
        return random_crop(crop, crop_size=self.sample_size)[0].squeeze(0)

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
                segmentation_sitk = sitk.ReadImage(segmentation_file)
                if self.sample_size is not None:
                    data_np = self._sample(data_sitk, segmentation_sitk)
                else:
                    data_np = np.expand_dims(sitk.GetArrayFromImage(data_sitk).transpose(), axis=0)
                data_batch.append(data_np)
                label_batch.append(label)
                segmentation = [data_sitk.TransformPhysicalPointToIndex(segmentation_sitk.GetOrigin()),
                                segmentation_sitk.GetSize()]
                segmentation_batch.append(segmentation)
        batch = {"data": np.stack(data_batch),
                 "label": np.stack(label_batch),
                 "segmentation": np.stack(segmentation_batch)}
        self._current_position += self.number_of_threads_in_multithreaded * self.batch_size
        return batch


def get_data_augmenter(data, batch_size=1, sample_size=None, transforms=None, threads=1):
    transforms = [] if transforms is None else transforms
    threads = min(int(np.ceil(len(data) / batch_size)), threads)
    loader = DataLoader(data=data, batch_size=batch_size, sample_size=sample_size,
                        number_of_threads_in_multithreaded=threads)
    transforms = transforms + [PrepareForTF()]
    return MultiThreadedAugmenter(loader, Compose(transforms), threads)


def get_tf_dataset(augmenter, input_shape):
    batch_size = augmenter.generator.batch_size

    def generator():
        augmenter.restart()
        for batch in augmenter:
            yield batch
        augmenter._finish()

    tf_dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32),
        tf.TensorSpec(shape=batch_size, dtype=tf.int32)))
    return tf_dataset


class PrepareForTF(AbstractTransform, ABC):
    def __call__(self, **data_dict):
        data = data_dict["data"]
        data = np.moveaxis(data, 1, -1)
        label = data_dict["label"]
        return tf.convert_to_tensor(data, dtype=tf.float32), tf.convert_to_tensor(label, dtype=tf.int32)
