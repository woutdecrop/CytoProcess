# CytoProcess

Package to process images and their features from .cyz files from the CytoSense and upload them to EcoTaxa.


## Installation

NB: As for all things Python, you should preferrably install CytoProcess within a Python venv/coda environment. The package is tested with Python=3.11 and should therefore work with this or a more recent version. To create a conda environment, use

```bash
conda create -n cytoprocess python=3.11
conda activate cytoprocess
```

Then install the sable version with

```bash
pip install cytoprocess
```

*or* the development version with

```bash
pip install git+https://github.com/jiho/cytoprocess.git
```

The Python package includes a command line tool, which should become available from within a terminal. To try it and output the help message

```bash
cytoprocess
```

CytoProcess depends on [Cyz2Json](https://github.com/OBAMANEXT/cyz2json). To install it, run

```bash
cytoprocess install
```


## Usage

CytoProcess uses the concept of "project". A project corresponds conceptually to a cruise, a time series, etc. Practically, it is a directory with a specific set of subdirectories that contain all files related to the cruise/time series/etc. It corresponds to a single EcoTaxa project.

Each .cyz file is considered as a "sample" (and will correspond to an EcoTaxa sample).

```
my_project/
    config      configuration files
    raw         source .cyz files
    converted   .json files converted from .cyz by Cyz2Json
    meta        files storing metadata and is mapping from .json to EcoTaxa
    images      images extracted from the .json files, in one subdirectory per file
    work        information extracted by the various processing steps (metadata, pulses, features, etc.)
    ecotaxa     .zip files ready for upload in EcoTaxa
    logs        logs of all commands executed on this project, per day
```

A CytoProcess command line looks like

```bash
cytoprocess --global-option command --command-option project_directory
```

To know which global options and which commands are available, use

```bash
cytoprocess --help
```

To know which options are available for a given command

```bash
cytoprocess command --help
```

### Creating and populating a project

Use

```bash
cytoprocess create path/to/my_project
```

Then copy/move the .cyz files that are relevant for this project in `my_project/raw`. If you have an archive of .cyz files organised differently, you should be able to symlink them in `my_project/raw` instead of copying them.


### Processing samples in a project

List available samples and create the `meta/samples.csv` file

```bash
cytoprocess list path/to/my_project
```

When the sample filename contains a timestamp such as `2025-09-04_08h06` or `2025-09-04%2008h06`, `cytoprocess list` fills `object_date` and `object_time` automatically in `meta/samples.csv`. These become the Date and Time fields in EcoTaxa object details and can be used for filtering.

Manually enter any remaining required metadata (such as lon, lat, depth, etc.) in the .csv file. You can add or remove columns as you see fit, you can use the option `--extra-fields` to determine which to add. The conventions follow those of EcoTaxa. Then perform all processing steps, for all samples, with default options:

```bash
cytoprocess all path/to/my_project
```

To run the same processing chain and also classify images, upload the samples, and sync predictions to EcoTaxa, use:

```bash
cytoprocess all_predict path/to/my_project
```

If you want to know the details, or proceed manually, the main steps are:

```bash
# convert .cyz files into .json and create a placeholder its metadata
cytoprocess convert path/to/project

# extract sample/acq/process level metadata from each .json file
cytoprocess extract_meta path/to/project
# extract cytometric features for each imaged particle
cytoprocess extract_cyto path/to/project
# compute pulse shapes polynomial summaries for each imaged particle
cytoprocess summarise_pulses path/to/project

# extract images and image features
cytoprocess extract_images path/to/project
# predict image classes with the classifier API
cytoprocess predict_images path/to/project

# prepare files for ecotaxa upload
cytoprocess prepare path/to/project
# upload them to EcoTaxa
cytoprocess upload path/to/project
# update existing EcoTaxa objects with predicted classes
cytoprocess overwrite_ecotaxa path/to/project
# upload new samples, then apply all available predictions
cytoprocess upload_all_predictions path/to/project
```

### Image extraction and segmentation fallback

`extract_images` uses the instrument background stored in the converted `.json` file to segment the object in each image. If a converted `.cyz` file contains no images, the sample is skipped cleanly: an empty `image_features.parquet` is written so the pipeline can continue without reprocessing that file forever.

Samples with no particles are handled the same way by the pulse and EcoTaxa preparation steps. Empty summary files and placeholder output directories are created where needed, and `prepare` skips the sample with a `No imaged particles` message instead of stopping the whole project.

If an image exists but no object can be segmented from the background, CytoProcess keeps the object instead of dropping it. It writes the raw image, writes a blank fallback mask, and records:

```text
object_segmentation_status = failed
```

During `predict_images`, these failed-segmentation objects are not sent to the classifier model. They are written directly to `predictions.parquet` with the EcoTaxa label:

```text
object_annotation_category = out of focus
object_annotation_category_id = 95471
object_annotation_probability = 1.0
```

This keeps the object visible in EcoTaxa while making it explicit that the image could not be segmented reliably.

### Prediction Upload Modes

`predict_images` creates one prediction file in `work/` for each sample:

```bash
work/<sample>/predictions.parquet
```

This file stores the top predicted label and, when available, the top 3 predicted labels and their scores. Failed-segmentation objects are included in the same file as `out of focus` and are not classified by the model.

Use the following commands depending on what is already present on EcoTaxa:

```bash
# Upload samples only, using the standard EcoTaxa ZIP import flow
cytoprocess upload path/to/project

# Apply predictions to objects that already exist in EcoTaxa
cytoprocess overwrite_ecotaxa path/to/project

# Rebuild ZIPs without prediction columns, upload them, then apply predictions
cytoprocess upload_all_predictions path/to/project
```

Notes:

- `upload` imports the prepared EcoTaxa ZIP files.
- `overwrite_ecotaxa` reads `work/<sample>/predictions.parquet` and applies the stored predictions to objects that already exist in EcoTaxa.
- `upload_all_predictions` uploads the sample data first, then applies all available predictions through the EcoTaxa API.
- EcoTaxa records these as automatic predictions, so the history `Author` field remains empty (`-`) even though the model name is stored in the exported prediction metadata.


### Customisation

To process a single sample, use

```bash
cytoprocess --sample 'name_of_cyz_file' command path/to/project
```

All commands will skip the processing of a given sample if the output is already present. To re-process and overwrite, use the `--force` option.

For metadata and cytometric features extraction (`extract_meta` and `extract_cyto`), information from the json file needs to be curated and translated into EcoTaxa metadata columns. This is defined in the configuration file, by `key: value` pairs of the form `json.fields.item.name: ecotaxa_name`. To get the list of possible json fields, use the `--list` option for `extract_meta` or `extract_cyto`; it will write a text file in `meta` with all possibilities. You can then copy-paste them to `config/config.yaml`.

Even with all these fields available, the CytoSense may not record relevant metadata such as latitude, longitude, and date of each sample, which EcoTaxa needs to filter the data or export it to other data bases. You can provide such fields manually by editing the `meta/samples.csv` file.


### Cleaning up after processing

Because everything is stored in the EcoTaxa files and can be re-generated from the .cyz files, you may want to remove the intermediate files, to reclaim disk space. This is done with

```bash
cytoprocess clean path/to/project
```

## Development

Fork this repository, clone your fork.

Prepare your development environment by installing the dependencies within a conda environment

```bash
conda create -n cytoprocess python=3.11
conda activate cytoprocess
pip install -e .
```

This creates a `cytoprocess.egg-info` directory at the root of the package's directory. It is safely ignored by git (and you should too).

Now, either run commands as you normally would

```bash
cytoprocess --help
```

or call the module explicitly

```bash
python -m cytoprocess --help
```

Any edits made to the files are immediately reflected in the output (because the package was installed in "editable" mode: `pip install -e ...` ; or is run directly as a module: `python -m ...`).
