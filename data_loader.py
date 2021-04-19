import logging
import os
from abc import ABC

import SimpleITK as sitk
import numpy as np

from batchgenerators.dataloading import SlimDataLoaderBase


class DataLoader(SlimDataLoaderBase, ABC):
    def __init__(self,
                 data_directory,
                 batch_size,
                 crop="none",
                 number_of_threads_in_multithreaded=4,
                 roi=None):
        super().__init__(None, batch_size, number_of_threads_in_multithreaded)
        assert crop in ["none", "fixed", "roi", "seg_bin", "seg"]
        if crop != "none":
            assert roi is not None
            self.roi_root = roi[:3]
            self.roi_size = roi[3:]
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
            # load files as sitk objects
            batch_sitk = []
            for i in range(self.batch_size):
                index = self.current_position + i
                if index < len(self._data):
                    patient_id, data_file, segmentation_file = self._data[self.current_position]
                    data_sitk = sitk.ReadImage(data_file, imageIO="NrrdImageIO")
                    segmentation_sitk = None
                    if self.crop != "none":
                        segmentation_sitk = sitk.ReadImage(segmentation_file, imageIO="NrrdImageIO")
                    batch_sitk.append((patient_id, data_sitk, segmentation_sitk))
            # preprocess data
            batch_np = []
            for patient_id, data_sitk, segmentation_sitk in batch_sitk:
                if self.crop == "none":
                    data_np = sitk.GetArrayFromImage(data_sitk)
                    data_np = np.expand_dims(data_np.transpose(), axis=0)
                else:
                    data_sitk_cropped = data_sitk[
                                        self.roi_root[0]:self.roi_root[0] + self.roi_size[0],
                                        self.roi_root[1]:self.roi_root[1] + self.roi_size[1],
                                        self.roi_root[2]:self.roi_root[2] + self.roi_size[2]]
                    data_np = sitk.GetArrayFromImage(data_sitk_cropped).transpose()
                    if self.crop in ["roi", "seg_bin", "seg"]:
                        segmentation_root = segmentation_sitk.TransformIndexToPhysicalPoint((0, 0, 0))
                        segmentation_root = data_sitk.TransformPhysicalPointToIndex(segmentation_root)
                        segmentation_root = np.array(segmentation_root) - np.array(self.roi_root)
                        assert all(x >= 0 for x in segmentation_root)
                        segmentation_size = segmentation_sitk.GetSize()
                        assert all(seg_root + seg_size <= roi_size for seg_root, seg_size, roi_size in
                                   zip(segmentation_root, segmentation_size, self.roi_size))
                        if self.crop == "roi":
                            mask = np.ones_like(data_np, dtype=np.bool)
                            mask[segmentation_root[0]:segmentation_root[0] + segmentation_size[0],
                                 segmentation_root[1]:segmentation_root[1] + segmentation_size[1],
                                 segmentation_root[2]:segmentation_root[2] + segmentation_size[2]] = False
                            data_np = np.expand_dims(np.where(mask, 0, data_np), axis=0)
                        else:
                            segmentation_np = sitk.GetArrayFromImage(segmentation_sitk).transpose()
                            if self.crop == "seg_bin":
                                segmentation_np = segmentation_np.sum(axis=0, keepdims=True)
                            data_layer = []
                            for mask in np.split(segmentation_np, indices_or_sections=segmentation_np.shape[0], axis=0):
                                padding = list((seg_root, roi_size - seg_size - seg_root)
                                               for seg_root, seg_size, roi_size in
                                               zip(segmentation_root, segmentation_size, data_np.shape))
                                mask = np.pad(mask.squeeze(), padding, constant_values=False)
                                data_layer.append(np.where(mask, data_np, 0))
                            data_np = np.stack(data_layer)
                batch_np.append(data_np)
            batch = {"data": np.stack(batch_np)}

            self.current_position += self.number_of_threads_in_multithreaded * self.batch_size
            return batch
        else:
            self.reset()
            raise StopIteration
