# 🌸 Iris Flower Classifier

A machine learning desktop application for classifying Iris flower species.  
Built with **scikit-learn** and **tkinter** — runs on Android (Pydroid 3) and any desktop Python environment.

---

## What It Does

This project trains five classification models on the classic Iris dataset and provides a graphical interface for both training and real-time prediction.

**Training pipeline (`train_model.py`)**

1. Loads `Iris.csv` and performs Exploratory Data Analysis (EDA)
2. Preprocesses data — drops the `Id` column, removes duplicates, encodes labels, splits 80/20, and scales features with `StandardScaler`
3. Trains five models: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, and Support Vector Machine
4. Evaluates each model with Accuracy, Precision, Recall, and F1-Score (weighted)
5. Selects the best model by F1-Score and saves it (along with the scaler and label encoder) to a `.pkl` file using `joblib`
6. Generates and saves five visualisation plots to disk

**GUI (`gui.py`)**

- Three-tab interface: **Train**, **Predict**, and **Output Log**
- Runs the full training pipeline in a background thread (UI stays responsive)
- Loads any previously saved `.pkl` model for prediction without retraining
- Accepts four flower measurements and returns the predicted species, confidence %, and per-class probabilities
- Includes sample quick-fill buttons for all three species
- Validates input ranges and warns on out-of-range values
- Live console output is streamed to the Output Log tab in real time

---

## Project Files

```
iris-classifier/
├── train_model.py      # ML pipeline — IrisClassifier class
├── gui.py              # tkinter GUI application
├── Iris.csv            # Dataset (required at runtime)
└── README.md
```

Files produced after training:

```
iris_classifier.pkl         # Best model + scaler + label encoder
class_distribution.png
feature_distributions.png
pair_plot.png
correlation_heatmap.png
box_plots.png
model_comparison.png
```

---

## Requirements

| Package        | Purpose                        |
|----------------|--------------------------------|
| `numpy`        | Numerical operations           |
| `pandas`       | Data loading and manipulation  |
| `scikit-learn` | ML models, preprocessing, metrics |
| `matplotlib`   | Plot generation                |
| `seaborn`      | Statistical visualisations     |
| `joblib`       | Model serialisation            |
| `tkinter`      | GUI framework (stdlib)         |

`tkinter` is part of the Python standard library and does not need to be installed separately on desktop. On Pydroid 3 it is included in the app.

### Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

Or from a requirements file:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
```

---

## Dataset

The project expects `Iris.csv` with the following columns:

| Column          | Type    | Description                  |
|-----------------|---------|------------------------------|
| `Id`            | int     | Row identifier (dropped during preprocessing) |
| `SepalLengthCm` | float   | Sepal length in centimetres  |
| `SepalWidthCm`  | float   | Sepal width in centimetres   |
| `PetalLengthCm` | float   | Petal length in centimetres  |
| `PetalWidthCm`  | float   | Petal width in centimetres   |
| `Species`       | string  | Target class (Iris-setosa / Iris-versicolor / Iris-virginica) |

The dataset contains **150 samples**, 50 per species.  
Source: [Kaggle — Iris Flower Dataset](https://www.kaggle.com/datasets/uciml/iris)

---

## Usage

### Running the GUI (recommended)

Place `gui.py`, `train_model.py`, and `Iris.csv` in the same folder, then run:

```bash
python gui.py
```

#### On Pydroid 3 (Android)
1. Install **Pydroid 3** from the Play Store
2. Open Pydroid 3's pip installer and install the packages listed above
3. Copy `gui.py`, `train_model.py`, and `Iris.csv` into the same directory
4. Open `gui.py` in Pydroid 3 and tap **Run**

#### On Desktop (VS Code, terminal, IDLE)

```bash
python gui.py
```

Or, if you have multiple Python versions:

```bash
python3 gui.py
```

---

### GUI Walkthrough

**Train tab**

1. Set the path to `Iris.csv` (defaults to the current directory)
2. Set the output path for the saved model (defaults to `iris_classifier.pkl`)
3. Toggle EDA output and plot saving as needed
4. Click **▶ Start Training Pipeline**
5. Switch to the **Output Log** tab to watch live training output
6. Results summary appears in the Train tab once complete

**Predict tab**

- If you have just trained a model, it is available in memory immediately — no need to load a file
- To use a previously saved model, enter the `.pkl` path and click **Load Model**
- Enter four measurements (in centimetres) or use a **Quick fill** button to pre-populate a sample
- Click **🔍 Predict Species** to see the predicted species, confidence %, and per-class probabilities

**Output Log tab**

All `print()` output from both training and prediction is streamed here in real time. Use **Clear Log** to reset it.

---

### Running the training script directly (no GUI)

```bash
python train_model.py
```

This calls `IrisClassifier.run_full_pipeline()`, which runs the complete pipeline and saves the model to `iris_classifier.pkl` in the current directory.

---

### Using `IrisClassifier` programmatically

```python
from train_model import IrisClassifier

# Train
clf = IrisClassifier(data_path="Iris.csv", random_state=42)
clf.run_full_pipeline(save_model_path="iris_classifier.pkl")

# Predict with the trained model (in memory)
result = clf.predict(
    sepal_length=5.1,
    sepal_width=3.5,
    petal_length=1.4,
    petal_width=0.2
)
print(result["predicted_species"])   # e.g. Iris-setosa
print(result["confidence"])          # e.g. 99.87

# Load a saved model and predict
model_data = IrisClassifier.load_model("iris_classifier.pkl")
result = clf.predict(5.1, 3.5, 1.4, 0.2, model_data=model_data)
```

---

## Models Trained

| Model                    | Parameters used                                      |
|--------------------------|------------------------------------------------------|
| Logistic Regression      | `max_iter=1000`, `multi_class='ovr'`                 |
| Decision Tree            | `max_depth=5`, `min_samples_split=5`                 |
| Random Forest            | `n_estimators=100`, `max_depth=5`                    |
| K-Nearest Neighbors      | `n_neighbors=5`, `weights='uniform'`                 |
| Support Vector Machine   | `kernel='rbf'`, `probability=True`, `C=1.0`          |

All models use `random_state=42` where applicable. The best model is selected by weighted F1-Score.

---

## Evaluation Metrics

Each model is evaluated on the held-out test set (20% of data, stratified split):

- **Accuracy** — fraction of correctly classified samples
- **Precision** — weighted average across classes
- **Recall** — weighted average across classes
- **F1-Score** — weighted harmonic mean of precision and recall
- **Confusion Matrix** — per-class prediction breakdown
- **Classification Report** — per-class precision, recall, and F1

---

## Visualisations Generated

| File                        | Content                                      |
|-----------------------------|----------------------------------------------|
| `class_distribution.png`    | Bar chart of sample counts per species       |
| `feature_distributions.png` | Histograms of all four features              |
| `pair_plot.png`             | Seaborn pair plot coloured by species        |
| `correlation_heatmap.png`   | Pearson correlation matrix of features       |
| `box_plots.png`             | Box plots of each feature grouped by species |
| `model_comparison.png`      | Grouped bar chart of all model metrics       |

---

## Saved Model Format

The `.pkl` file is a dictionary serialised with `joblib.dump` and contains:

| Key             | Content                                      |
|-----------------|----------------------------------------------|
| `model`         | Trained scikit-learn estimator               |
| `model_name`    | String name of the best model                |
| `scaler`        | Fitted `StandardScaler`                      |
| `label_encoder` | Fitted `LabelEncoder`                        |
| `feature_names` | `['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']` |
| `random_state`  | `42`                                         |

---

## Project Structure Details

### `train_model.py` — `IrisClassifier` class

| Method                  | Description                                            |
|-------------------------|--------------------------------------------------------|
| `load_data()`           | Reads `Iris.csv` into a DataFrame                      |
| `explore_data()`        | Prints EDA statistics to stdout                        |
| `visualize_data()`      | Generates and saves all six plots                      |
| `preprocess_data()`     | Cleans, encodes, splits, and scales the data           |
| `initialize_models()`   | Creates the five model instances                       |
| `train_models()`        | Fits all models on the training set                    |
| `evaluate_models()`     | Computes and prints metrics for each model             |
| `compare_models()`      | Prints a comparison table and identifies the best model |
| `plot_model_comparison()` | Saves the model comparison bar chart               |
| `save_model()`          | Persists the best model bundle to disk                 |
| `load_model()`          | Static method — loads a saved model bundle             |
| `predict()`             | Returns species prediction and probabilities           |
| `run_full_pipeline()`   | Runs all steps end-to-end                              |

### `gui.py` — `IrisApp` class

Built with `tkinter` and `tkinter.ttk`. Uses `threading.Thread` to run training without blocking the UI. Redirects `sys.stdout` and `sys.stderr` to the Output Log tab so all print output from `train_model.py` appears in the GUI.

---

## Troubleshooting

**`FileNotFoundError: Dataset not found at: Iris.csv`**  
Make sure `Iris.csv` is in the same folder as `gui.py` and `train_model.py`, or use the Browse button to select its full path.

**`ModuleNotFoundError: No module named 'sklearn'`** (or similar)  
Run `pip install scikit-learn` (and the other packages listed in Requirements).

**Plots not appearing on Pydroid 3**  
The scripts save plots to disk rather than displaying them with `plt.show()`. Check the folder where the model was saved for `.png` files.

**`tkinter` not available**  
On some Linux distributions, tkinter is a separate package. Install it with:
```bash
sudo apt-get install python3-tk   # Debian/Ubuntu
```
On Pydroid 3, tkinter is bundled and requires no extra installation.

---

## License

MIT License — free to use and modify.
