## Risk prediction for pancreatic fistula after pancreatoduodenectomy using convolutional neural networks

### 1. Preprocess data

Run the script (see scripts/dataset):

`preprocess_dataset.py --input path/to/raw_data --output output/dir --config path/to/preprocess.config`

To adjust the script to the directory structure of the dataset, the 3 global
variables `PATIENT_ID_PATTERN, DATA_FILE_PATTERN` and `SEGMENTATION_FILE_PATTERN`, declared at the top of the script,
can be changed. A default config file can be found in the same directory. The script itself does 3 things:

1. Resample the CT series to a common spacing and calculate the individual region of interests. The latter, by default,
   is done using the smallest bounding box surrounding the annotated segmentation. This can be changed by for example
   adding a margin to the RoI or set a fixed aspect ratio (might make sense when resizing the different RoIs to a
   common size later).
2. Crop the CTs using a common specified bounding box. There are 3 types of crops:
    - `fixed:` the CT is simply cropped using the bb with the RoI in the center
    - `roi:` the same as fixed, but everything outside the RoI is masked
    - `seg:` the same as fixed, but everything outside the annotated segmentation is masked
3. Normalize the CT's intensities to a specifed range e.g. [0, 1]

After the script finished, the directory for one patient contains (can be less when changing `tasks` or `crops`
parameters in the config file):

- a `raw` directory with the original resampled CT + segmentation and all the crops without normalization (these will
  not be used during training unless online normalization is used)
- the input ready crops (have the same size for all patients)
    - `fixed.nrrd:`the fixed crop
    - `roi.nrrd:` the roi crop
    - `seg.nrrd:` the seg crop
- the CT specific crops (used when choosing `sample` or `resize` as input type during training)
    - `roi_only.nrrd:` only the RoI calculated in the first step
    - `roi_only_masked.nrrd:` the same as above but everything outside the annotated segmentation is masked

Preprocessing can take quite some time and might require a lot of disk space!

### (1.1 Analyze data)

The two scripts `analyze_dataset.py` and `visualize_dataset.py` can be used to calculate and visualize some properties
of the dataset to find the optimal bounding box and normalization ranges. Therefore, first run

`analyze_dataset.py --data path/to/preprocessed/data --output output/directory`

The script collects some data and generates an `analysis_results.json` in the output directory. This file is then used
by the second script:

`visualize_dataset.py --file path/to/analysis_results.json --hist_coverage <decimal> --f_beta <decimal>`

The `--hist_coverage` parameter is a float value in the range [0, 1]. This is the percentage of intensities covered when
using the calculated ranges for normalization. A lower value means a smaller range --> more intensities are clipped to
that range before normalization. By clipping intensities before normalization, outliers can be removed from the
distribution.

The `--f_beta` is a float value in the range [0, inf]. This is the beta parameter for the F-beta score controlling the
trade off between recall and precision. In this context it is used to calculate an optimal bounding box used for
cropping. For a given bounding box, precision here means: the average ratio of RoI to non-RoI for all CT’s and recall
analogically is the average coverage of RoI for all CT's. The problem is then to find a bounding which maximizes the
average F-beta score.

### 2 Training

Run the script:

`train.py --config path/to/train.config`

See the `default.config` for available parameters.

