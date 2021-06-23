import csv
import os
import random
from abc import ABC
from enum import IntEnum

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop
from batchgenerators.augmentations.spatial_transformations import augment_resize

from batchgenerators.dataloading import SlimDataLoaderBase, MultiThreadedAugmenter
from batchgenerators.transforms import AbstractTransform, Compose

from augmentation import get_transforms
from scripts.dataset.preprocess_dataset import normalize, HOUNSFIELD_BOUNDARIES


def scan_data_directory(data_directory, crop="none", normalized=True, blacklist=None):
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
                data_file = os.path.join(directory.path, f"{crop}.nrrd") if normalized else \
                    os.path.join(directory.path, "raw", f"{crop}.nrrd")
                segmentation_file = os.path.join(directory.path, "raw", "segmentation.seg.nrrd")
                data.append((label, data_file, segmentation_file))
            else:
                data.append(directory.path)
    return data


def get_dataset_from_config(config, volumetric=True):
    if "data" in config:
        config = config["data"]
    masked = config["masked"]
    if config["input_type"] == "crop":
        crop = "seg" if masked else config["crop"]["type"]
    elif config["input_type"] == "sample" or config["input_type"] == "resize":
        crop = "roi_only_masked" if masked else "roi_only"
    else:
        raise ValueError(f"Invalid 'input_type': {config['input_type']}")
    normalized = not ("online_normalization" in config and config["online_normalization"])
    dataset = scan_data_directory(config["path"], crop=crop, normalized=normalized, blacklist=config["blacklist"])
    if config["input_type"] == "sample" or config["input_type"] == "resize":
        input_shape = [*config[config["input_type"]]["size"], 1]
        if not volumetric and len(input_shape) == 4:
            input_shape = [*input_shape[:2], 1]
    else:
        dummy_loader = get_data_augmenter(dataset[:1], volumetric=volumetric)
        input_shape = next(dummy_loader)[0].shape[-4:]
        dummy_loader._finish()
    return dataset, input_shape


def visualize_data(data: np.ndarray):
    import matplotlib.pyplot as plt
    if data.ndim == 5:
        data = data[1, :, :, :, :]
    data = data.squeeze()
    if len(data.shape) == 3:
        for image in np.split(data, data.shape[-1], -1):
            if np.max(image) > 0:
                image = np.moveaxis(image, 0, 1)
                plt.imshow(image)
                plt.show()
    else:
        image = np.moveaxis(data, 0, 1)
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
    class Mode(IntEnum):
        NORMAL = 0
        SAMPLE = 1
        RESIZE = 2

    def __init__(self,
                 data,
                 batch_size,
                 volumetric=True,
                 mode=Mode.NORMAL,
                 balance=False,
                 normalization_range=None,
                 vector_generator=None,
                 input_shape=None,
                 sample_count=1,
                 number_of_threads_in_multithreaded=4,
                 epochs=1,
                 seed=None):
        super().__init__(None, batch_size, number_of_threads_in_multithreaded)
        self.volumetric = volumetric
        if not self.volumetric:
            self._data = []
            for label, ct, seg in data:
                slices = sitk.ReadImage(ct).GetSize()[2]
                for i in range(slices):
                    self._data.append((label, (ct, i), seg))
        else:
            self._data = data
        self.mode = mode
        self.normalization_range = normalization_range
        self.vector_generator = vector_generator
        self.input_shape = input_shape
        if self.mode == DataLoader.Mode.SAMPLE:
            self._data *= sample_count
        if balance:
            trues = list(x for x in self._data if x[0] == 1)
            falses = list(set(self._data) - set(trues))
            majority = trues if len(trues) > len(falses) else falses
            minority = trues if len(trues) <= len(falses) else falses
            majority = random.sample(majority, len(minority))
            self._data = minority + majority

        self.batch_size = _fit_batch_size(len(self._data), self.batch_size)
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
        vector_batch = None if self.vector_generator is None else []
        label_batch = []
        loaded_cts = {}
        for i in range(self.batch_size):
            index = self._current_position + i
            if index < len(self._data):
                label, data, segmentation_file = self._data[index]
                data_file = data if self.volumetric else data[0]
                slice = None if self.volumetric else data[1]
                if data_file in loaded_cts:
                    data_np = loaded_cts[data_file]
                else:
                    data_sitk = sitk.ReadImage(data_file)
                    data_np = np.expand_dims(sitk.GetArrayFromImage(data_sitk).transpose(), axis=0)
                    if self.normalization_range is not None:
                        data_np = normalize(data_np, self.normalization_range, [0, 1])
                    loaded_cts[data_file] = data_np
                if slice is not None:
                    data_np = data_np[:, :, :, slice]
                vector_gen_args = {"input_shape_pre": data_np.shape[-3:]}
                if self.mode == DataLoader.Mode.RESIZE:
                    data_np = augment_resize(data_np, None, self.input_shape)[0].squeeze(0)
                    data_np = np.expand_dims(data_np, 0)
                    vector_gen_args["input_shape_post"] = self.input_shape
                elif self.mode == DataLoader.Mode.SAMPLE:
                    data_np = np.expand_dims(data_np, 0)
                    with RNGContext(self.seed):
                        data_np = random_crop(data_np, crop_size=self.input_shape)[0].squeeze(0)
                    vector_gen_args["input_shape_post"] = self.input_shape
                if self.vector_generator is not None:
                    vector_batch.append(self.vector_generator(**vector_gen_args))
                data_batch.append(data_np)
                label_batch.append(label)
        batch = {"data": np.stack(data_batch),
                 "vector": None if vector_batch is None else np.stack(vector_batch),
                 "label": np.stack(label_batch)}
        self._current_position += self.number_of_threads_in_multithreaded * self.batch_size
        return batch


def get_data_augmenter(data, batch_size=1,
                       mode=DataLoader.Mode.NORMAL,
                       volumetric=True,
                       balance=False,
                       normalization_range=None,
                       vector_generator=None,
                       input_shape=None,
                       sample_count=1,
                       transforms=None,
                       threads=1,
                       seed=None):
    transforms = [] if transforms is None else transforms
    threads = min(int(np.ceil(len(data) / batch_size)), threads)
    loader = DataLoader(data=data,
                        batch_size=batch_size,
                        mode=mode,
                        balance=balance,
                        volumetric=volumetric,
                        normalization_range=normalization_range,
                        vector_generator=vector_generator,
                        input_shape=input_shape,
                        sample_count=sample_count,
                        number_of_threads_in_multithreaded=threads,
                        seed=seed)
    transforms = transforms + [PrepareForTF()]
    return MultiThreadedAugmenter(loader, Compose(transforms), threads)


def get_tf_dataset(augmenter, input_shapes):
    batch_size = augmenter.generator.batch_size

    def generator():
        augmenter.restart()
        for batch in augmenter:
            yield batch
        augmenter._finish()

    input_signatures = tuple([tf.TensorSpec(shape=(None, *shape), dtype=tf.float32) for shape in input_shapes])
    tf_dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        input_signatures,
        tf.TensorSpec(shape=batch_size, dtype=tf.int32)))
    return tf_dataset


class PrepareForTF(AbstractTransform, ABC):
    def __call__(self, **data_dict):
        data = data_dict["data"]
        data = np.moveaxis(data, 1, -1)
        label = data_dict["label"]
        inputs = [tf.convert_to_tensor(data, dtype=tf.float32)]
        if data_dict["vector"] is not None:
            inputs.append(tf.convert_to_tensor(data_dict["vector"], dtype=tf.float32))
        return tuple(inputs), tf.convert_to_tensor(label, dtype=tf.int32)


def _fit_batch_size(data_count, max_batch_size):
    for i in range(max_batch_size, 0, -1):
        if data_count % i == 0:
            return i


def get_resize_ratio(**kwargs):
    pre = np.array(kwargs["input_shape_pre"])
    post = np.array(kwargs["input_shape_post"])
    return pre / post


def get_data_loader_from_config(train_data, validation_data, config, ct_shape, volumetric=True):
    config = config.get("data", config)
    input_type = config["input_type"]
    sample_count = 1
    vector_shape = None
    vector_generator = None
    if input_type == "crop":
        mode = DataLoader.Mode.NORMAL
    elif input_type == "sample":
        mode = DataLoader.Mode.SAMPLE
        sample_count = config["sample"]["count"]
    elif input_type == "resize":
        mode = DataLoader.Mode.RESIZE
        if config["resize"]["use_ratio_vector"]:
            vector_shape = (3,)
            vector_generator = get_resize_ratio
    else:
        raise ValueError(f"Invalid 'input_type': {config['input_type']}")
    normalization_range = HOUNSFIELD_BOUNDARIES if ("online_normalization" in config and
                                                    config["online_normalization"]) else None
    balance = config.get("balance", False)
    train_augmenter = get_data_augmenter(data=train_data,
                                         batch_size=config["batch_size"],
                                         mode=mode,
                                         balance=balance,
                                         volumetric=volumetric,
                                         normalization_range=normalization_range,
                                         vector_generator=vector_generator,
                                         input_shape=ct_shape[:-1],
                                         sample_count=sample_count,
                                         transforms=get_transforms(),
                                         threads=config["loader_threads"])
    input_shape = [ct_shape]
    if vector_shape is not None:
        input_shape.append(vector_shape)
    train_dl = get_tf_dataset(train_augmenter, input_shape)
    #   cache validation data
    val_augmenter = get_data_augmenter(data=validation_data,
                                       batch_size=config["batch_size"],
                                       mode=mode,
                                       balance=balance,
                                       volumetric=volumetric,
                                       normalization_range=normalization_range,
                                       vector_generator=vector_generator,
                                       input_shape=ct_shape[:-1],
                                       sample_count=sample_count,
                                       seed=42)
    val_dl = get_tf_dataset(val_augmenter, input_shape)
    return train_dl, val_dl, input_shape
