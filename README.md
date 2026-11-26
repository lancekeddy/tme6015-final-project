# Load-shape archetypes from monthly electricity bills

TME 6015 (Machine Learning) final project. The task is to classify a commercial building's load-shape archetype from what a utility bill shows: 12 monthly energy totals (kWh), 12 monthly billed peaks (kW), and basic metadata (floor area, latitude, building type, region, meter interval). There are four classes: `flat_7day`, `daytime_7day`, `daytime_weekday` and `weekday_moderate`. The labels are assigned by fixed rules from each building's interval meter data (EnerNOC 2012, 5-minute; Building Data Genome Project 2, hourly), and the bills are built from the same interval data, so the models learn to recover a shape that the bills alone do not show directly. Two Keras models are trained: a dense neural network and a 1-D convolutional network over the 12-month sequence. The intended downstream use is a light bill-screening step that flags buildings likely to be "shaveable" (`daytime_weekday` or `weekday_moderate`) for peak-shaving follow-up.

## Repository layout

```
.
├── README.md                  this file
├── LICENSE                    MIT licence (code)
├── LICENSE-NOTES.md           data licence terms and attribution
├── requirements.txt           Python packages
├── src/
│   ├── make_dataset_v2.py     builds the processed dataset from the raw sources
│   └── predict.py             command-line and Python inference for one building
├── data/
│   ├── README.md              description of every processed file and column
│   └── processed_v2/          processed dataset (dataset.csv, dataset_aug.csv, sites.csv, numbers.json)
├── notebooks/
│   ├── README.md
│   └── 01_models.ipynb        training and evaluation notebook
├── models/
│   ├── README.md
│   ├── dense_tuned.keras      trained dense network
│   ├── cnn_tuned.keras        trained 1-D CNN
│   └── preprocessing.json     feature order, training mean and std, classes, thresholds
└── examples/
    └── example_building.json  one building for predict.py
```

`data/raw/` is not in the repository (about 600 MB); see [Rebuild the dataset](#rebuild-the-dataset).

## Data sources

The processed dataset is derived from two public datasets:

1. **EnerNOC open dataset**: anonymized 5-minute energy data for 100 commercial and industrial sites, 2012. The licence file supplied with the data states: "All data is Copyright EnerNOC and is being made available for non-commercial and research purposes only under the Creative Commons Attribution-NonCommercial 3.0 license" (https://creativecommons.org/licenses/by-nc/3.0/). Dataset README: https://open-enernoc-data.s3.amazonaws.com/anon/README. Reading values are linearly scaled and latitude, longitude and floor area are anonymized by the source.
2. **Building Data Genome Project 2 (BDG2)**: hourly meter data for 1,636 non-residential buildings, 2016 and 2017. See the source repository for terms. Repository: https://github.com/buds-lab/building-data-genome-project-2. Citation: Miller, C., Kathirgamanathan, A., Picchetti, B. et al. The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition. *Scientific Data* 7, 368 (2020). https://doi.org/10.1038/s41597-020-00712-x

The derived dataset in `data/processed_v2/` combines both sources. It is shared for non-commercial academic use with attribution to EnerNOC and to the BDG2 authors as above. See [LICENSE-NOTES.md](LICENSE-NOTES.md). The code is under the MIT licence ([LICENSE](LICENSE)).

## Setup

Python 3.10 or later. The models in `models/` were trained in Google Colab on a T4 GPU (Python 3.13, TensorFlow 2.21, Keras 3.13, scikit-learn 1.6); `src/predict.py` has also been run with Keras 3.15.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Or open `notebooks/01_models.ipynb` in Google Colab, which already has every package in `requirements.txt`.

## Rebuild the dataset

The processed files are already in `data/processed_v2/`, so this step is only needed to reproduce them.

1. Download the raw data into `data/raw/` with this layout:

   ```
   data/raw/
   ├── EnerNOC csv-only/
   │   ├── meta/all_sites.csv
   │   └── csv/<site_id>.csv            100 files
   └── bdg2/
       ├── metadata.csv
       └── electricity_cleaned.csv
   ```

   - EnerNOC: download `csv-only.tar.gz` from the EnerNOC open dataset (listed in the README at https://open-enernoc-data.s3.amazonaws.com/anon/README), extract it, and name the extracted folder `EnerNOC csv-only`.
   - BDG2: from https://github.com/buds-lab/building-data-genome-project-2, take `data/metadata/metadata.csv` and `data/meters/cleaned/electricity_cleaned.csv`. The CSV files in that repository are stored with Git LFS, so clone it with Git LFS installed (or download each file from its GitHub page).

2. From the repository root, run:

   ```bash
   python src/make_dataset_v2.py
   ```

   It takes a few minutes on a laptop (EnerNOC about 25 s, BDG2 about 2 min) and uses a fixed seed (42).

3. Expected outputs in `data/processed_v2/` (the counts below are from the files in this repository and `numbers.json`):

   | file | rows | one row is |
   |---|---|---|
   | `sites.csv` | 2,387 | one building-year after duplicate removal (2,315 BDG2, 72 EnerNOC; 1,461 unique buildings; 30 duplicate meters dropped) |
   | `dataset.csv` | 2,191 | one building-year with 12 complete monthly bills (1,322 unique buildings) |
   | `dataset_aug.csv` | 21,896 | one building-year at one of 10 billing-cycle start offsets (0, 3, ..., 27 days) |
   | `numbers.json` | | summary counts, label thresholds and the k-means check |

   The script also writes `bills.csv`, `profiles_weekday_hourly.csv`, `profiles_weekend_hourly.csv` and `sites_all_with_duplicates.csv` to the same folder, and figures to `figures_v2/`. Those files are not included in the repository. Column descriptions are in [data/README.md](data/README.md).

## Training

Run `notebooks/01_models.ipynb` top to bottom, either locally (from the `notebooks/` folder, after `pip install -r requirements.txt`) or in Google Colab.

- Locally, the notebook reads `../data/processed_v2/dataset_aug.csv`.
- In Colab, upload `dataset_aug.csv` to the Colab working folder, or place it in Google Drive at `MyDrive/TME6015/dataset_aug.csv` and the notebook mounts Drive and reads it from there.

The notebook splits buildings (not rows) into train, validation and test sets, trains the dense and 1-D CNN models, runs cross-validation and learning curves, tunes the shaveable threshold on validation data, and prints a test-set summary. Its last section saves the two final models and the preprocessing file to `models/` (`dense_tuned.keras`, `cnn_tuned.keras`, `preprocessing.json`) and reloads the saved model to check it reproduces the same prediction. In Colab, zip the `models/` folder and download it from the Files panel.

## Inference

`src/predict.py` loads a saved model and `models/preprocessing.json`, rebuilds the same 55 features the notebook uses, and prints the class probabilities, the predicted archetype, `p_shaveable` (the probability of `daytime_weekday` plus `weekday_moderate`) and the shaveable decision at the model's tuned threshold.

From the repository root:

```bash
python src/predict.py examples/example_building.json                                    # 1-D CNN, the final model (default)
python src/predict.py examples/example_building.json --model models/dense_tuned.keras   # dense network
```

The input is a JSON file with these keys (see `examples/example_building.json`):

| key | meaning |
|---|---|
| `kwh` | 12 monthly energy values, kWh, January to December |
| `peak_kw_billed` | 12 monthly billed peak demand values, kW |
| `sq_ft` | floor area, square feet |
| `lat` | latitude, degrees |
| `resolution_min` | meter interval the bills come from: 15 or 60 minutes |
| `building_type` | one of the training building types, for example `office_commercial` |
| `region` | one of the training regions, for example `US_Mountain` |

The allowed `building_type` and `region` values are the `type_*` and `region_*` entries in `preprocessing.json`; an unknown value prints a warning and all of its one-hot columns are set to 0.

From Python:

```python
import json, sys
sys.path.insert(0, "src")
from predict import load, predict_building

model, prep = load("models/cnn_tuned.keras", "models/preprocessing.json")
building = json.load(open("examples/example_building.json"))
print(predict_building(building, model, prep, "cnn_tuned"))
```

`predict_building` returns a dictionary with `probabilities`, `predicted_archetype`, `p_shaveable`, `threshold` and `shaveable`.

## Results

| model | 4-fold CV balanced accuracy | 4-fold CV shaveable recall | 4-fold CV shaveable precision | tuned threshold | test shaveable recall, all rows | test shaveable precision, all rows | test shaveable recall, confident rows | test shaveable precision, confident rows |
|---|---|---|---|---|---|---|---|---|
| `cnn_tuned` | 0.607 +/- 0.033 | 0.843 +/- 0.018 | 0.764 +/- 0.029 | 0.306 | 0.923 | 0.693 | 0.976 | 0.874 |
| `dense_tuned` | 0.601 +/- 0.022 | 0.819 +/- 0.018 | 0.785 +/- 0.024 | 0.250 | 0.923 | 0.700 | 0.988 | 0.913 |

- Cross-validation columns: mean +/- standard deviation over 4 folds of the training and validation buildings (the cross-validated comparison in the notebook); shaveable recall and precision there use the most probable class (argmax).
- Chance level for 4 classes: balanced accuracy 0.25.
- Test columns: the held-out test split (441 rows, 264 buildings), shaveable = `p_shaveable` >= the tuned threshold chosen on validation data. Confident rows are test rows with `label_margin` >= 0.1 (204 rows).
- Run environment: Google Colab with a T4 GPU, Python 3.13.16, TensorFlow 2.21.0, Keras 3.13.2, scikit-learn 1.6.1, NumPy 2.4.6, pandas 2.2.3.

Full results, figures and per-model tables are in `notebooks/01_models.ipynb`.

## Generative AI use

Generative AI was used to assist in developing the code in this repository, as permitted by the course policy on code assistance. The full statement is in the appendix of the project report.

## Author

Lance Keddy, MTME program, University of New Brunswick
