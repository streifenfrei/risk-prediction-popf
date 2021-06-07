import csv
import os
from abc import ABC

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop
from batchgenerators.augmentations.spatial_transformations import augment_resize

from batchgenerators.dataloading import SlimDataLoaderBase, MultiThreadedAugmenter
from batchgenerators.transforms import AbstractTransform, Compose


def scan_data_directory(data_directory, crop="none", blacklist=None):
    assert crop in ["full", "fixed", "roi", "seg", "roi_only", "roi_only_masked", "all"]
    if blacklist is None:
        blacklist = []
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
            if patient_id in blacklist:
                continue
            label = labels[patient_id]

            if crop != "all":
                data_file = os.path.join(directory.path, f"{crop}.nrrd")
                segmentation_file = os.path.join(directory.path, "raw", "segmentation.seg.nrrd")
                data.append((label, data_file, segmentation_file))
            else:
                data.append(directory.path)
    return data


def visualize_data(data: np.ndarray):
    import matplotlib.pyplot as plt
    data = data.squeeze()
    for image in np.split(data, data.shape[-1], -1):
        plt.imshow(image)
        plt.show()


class RNGContext:
    def __init__(self, seed):
        self.seed = seed
        self.rng_state = None

    def __enter__(self):
        if self.seed is not None:
            self.rng_state = np.random.get_state()
            np.random.seed(seed=self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seed is not None:
            np.random.set_state(self.rng_state)


class DataLoader(SlimDataLoaderBase, ABC):
    def __init__(self,
                 data,
                 batch_size,
                 sample_size=None,
                 sample_count=1,
                 resize_to_sample_size=False,
                 number_of_threads_in_multithreaded=4,
                 epochs=1,
                 seed=None):
        super().__init__(None, batch_size, number_of_threads_in_multithreaded)
        self._data = data
        self.sample_size = sample_size
        self.resize_to_sample_size = resize_to_sample_size
        if self.sample_size is not None:
            self._data *= sample_count
        self._current_position = 0
        self.was_initialized = False
        self.epochs = epochs
        self._current_epoch = 0
        self.seed = seed

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
        for i in range(self.batch_size):
            index = self._current_position + i
            if index < len(self._data):
                label, data_file, segmentation_file = self._data[index]
                data_sitk = sitk.ReadImage(data_file)
                data_np = np.expand_dims(sitk.GetArrayFromImage(data_sitk).transpose(), axis=0)
                if self.sample_size is not None:
                    data_np = np.expand_dims(data_np, 0)
                    if self.resize_to_sample_size:
                        data_np = augment_resize(data_np, None, self.sample_size)[0].squeeze(0)
                    else:
                        with RNGContext(self.seed):
                            data_np = random_crop(data_np, crop_size=self.sample_size)[0].squeeze(0)
                data_batch.append(data_np)
                label_batch.append(label)
        batch = {"data": np.stack(data_batch),
                 "label": np.stack(label_batch)}
        self._current_position += self.number_of_threads_in_multithreaded * self.batch_size
        return batch


def get_data_augmenter(data, batch_size=1, sample_size=None, sample_count=1, transforms=None, threads=1, seed=None):
    transforms = [] if transforms is None else transforms
    threads = min(int(np.ceil(len(data) / batch_size)), threads)
    loader = DataLoader(data=data, batch_size=batch_size, sample_size=sample_size,
                        sample_count=sample_count, number_of_threads_in_multithreaded=threads, seed=seed)
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
