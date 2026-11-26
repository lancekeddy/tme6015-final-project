# Models

The last section of `notebooks/01_models.ipynb` writes three files here. They are what `src/predict.py` needs to run inference without retraining.

| file | contents |
|---|---|
| `dense_tuned.keras` | the final dense neural network (Keras 3 format). Input: one vector of the 55 standardized features. Output: 4 class probabilities (softmax). |
| `cnn_tuned.keras` | the final 1-D CNN (Keras 3 format). Two inputs: (A) a 12-month by 2-channel sequence of the standardized monthly kWh share and relative peak, and (B) the other 31 standardized features. Output: 4 class probabilities (softmax). |
| `preprocessing.json` | everything needed to rebuild the inputs: `feature_columns` (the 55 feature names in model order), `train_mean` and `train_std` (training-set statistics used for standardization), `classes` (class names in output order), `shaveable` (the classes counted in `p_shaveable`), `seq_cols` and `rest_cols` (which features go to CNN inputs A and B), and `thresholds` (each model's tuned `p_shaveable` threshold, chosen on validation data). |

Load a model with `keras.models.load_model("models/dense_tuned.keras")`, or use `src/predict.py` (see the main README).
