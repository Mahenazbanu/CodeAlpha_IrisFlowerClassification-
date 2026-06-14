# 🌸 Iris Flower Classification

A machine learning project that trains and compares five classification models on the classic Iris dataset and provides a **Gradio web interface** for real-time species prediction.

---

## What This Project Does

**Training pipeline (`train_model.py`)**

1. Loads `Iris.csv` and prints Exploratory Data Analysis (EDA) statistics to the console
2. Preprocesses the data — drops the `Id` column, removes duplicates, encodes labels with `LabelEncoder`, splits 80/20 with stratification, and scales features with `StandardScaler`
3. Trains five classification models: Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, and Support Vector Machine
4. Evaluates each model on the held-out test set using Accuracy, Precision, Recall, and F1-Score (all weighted)
5. Selects the best model by weighted F1-Score and saves it — together with the fitted scaler and label encoder — to a `.pkl` file using `joblib`
6. Generates and saves five visualisation plots to disk

**Web interface (`gui.py`)**

- Built with **Gradio** — runs in a browser at `http://localhost:7860`
- Accepts four flower measurements (in cm) and returns the predicted species, confidence %, and per-class probabilities
- Includes input validation with biological range checks
- Requires the `.pkl` model file produced by `train_model.py` to be present before launching

---

## Project Files

```
CodeAlpha_IrisFlowerClassification-/
├── train_model.py      # ML pipeline — IrisClassifier class
├── gui.py              # Gradio web application — IrisPredictor class
├── Iris.csv            # Dataset (required at runtime)
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT licence
└── README.md
```

Files produced after running `train_model.py`:

```
iris_classifier.pkl         # Best model + scaler + label encoder (joblib dict)
class_distribution.png
feature_distributions.png
pair_plot.png
correlation_heatmap.png
box_plots.png
model_comparison.png
```

---

## Requirements

- Python 3.8 or later

| Package        | Minimum version | Purpose                           |
|----------------|-----------------|-----------------------------------|
| `pandas`       | 2.0.0           | Data loading and manipulation     |
| `numpy`        | 1.24.0          | Numerical operations              |
| `matplotlib`   | 3.7.0           | Plot generation                   |
| `seaborn`      | 0.12.0          | Statistical visualisations        |
| `scikit-learn` | 1.3.0           | ML models, preprocessing, metrics |
| `gradio`       | 4.0.0           | Web interface                     |
| `joblib`       | 1.3.0           | Model serialisation               |

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

The project expects `Iris.csv` in the same directory as the scripts. The file must have these columns:

| Column          | Type   | Description                                                        |
|-----------------|--------|--------------------------------------------------------------------|
| `Id`            | int    | Row identifier — dropped during preprocessing                      |
| `SepalLengthCm` | float  | Sepal length in centimetres                                        |
| `SepalWidthCm`  | float  | Sepal width in centimetres                                         |
| `PetalLengthCm` | float  | Petal length in centimetres                                        |
| `PetalWidthCm`  | float  | Petal width in centimetres                                         |
| `Species`       | string | Target class: `Iris-setosa`, `Iris-versicolor`, or `Iris-virginica` |

The dataset contains 150 samples, 50 per species.  
Source: [Kaggle — UCI Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)

---

## Usage

### Step 1 — Train the models

```bash
python train_model.py
```

This runs the full pipeline and saves `iris_classifier.pkl` and the six plot files to the current directory. All EDA output and evaluation metrics are printed to the console.

You can also use the `IrisClassifier` class directly:

```python
from train_model import IrisClassifier

clf = IrisClassifier(data_path="Iris.csv", random_state=42)
clf.run_full_pipeline(save_model_path="iris_classifier.pkl")

# Predict with the in-memory trained model
result = clf.predict(
    sepal_length=5.1,
    sepal_width=3.5,
    petal_length=1.4,
    petal_width=0.2
)
print(result["predicted_species"])  # e.g. Iris-setosa
print(result["confidence"])         # e.g. 99.87

# Or load a saved model and predict
model_data = IrisClassifier.load_model("iris_classifier.pkl")
result = clf.predict(5.1, 3.5, 1.4, 0.2, model_data=model_data)
```

### Step 2 — Launch the web interface

**`iris_classifier.pkl` must exist before running this step.**

```bash
python gui.py
```

Open `http://localhost:7860` in your browser. Enter four measurements and click **Predict Species**.

> The Gradio server binds to `0.0.0.0:7860` with `share=False`. To create a temporary public link, set `share=True` in `gui.py`'s `demo.launch()` call.

---

## Models Trained

| Model                  | Parameters used                              |
|------------------------|----------------------------------------------|
| Logistic Regression    | `max_iter=1000`, `multi_class='ovr'`         |
| Decision Tree          | `max_depth=5`, `min_samples_split=5`         |
| Random Forest          | `n_estimators=100`, `max_depth=5`            |
| K-Nearest Neighbors    | `n_neighbors=5`, `weights='uniform'`         |
| Support Vector Machine | `kernel='rbf'`, `probability=True`, `C=1.0` |

All models use `random_state=42` where applicable. The best model is selected by weighted F1-Score on the test set.

---

## Evaluation Metrics

Each model is evaluated on the held-out test set (20% of the data, stratified split):

- **Accuracy** — fraction of correctly classified samples
- **Precision** — weighted average across the three classes
- **Recall** — weighted average across the three classes
- **F1-Score** — weighted harmonic mean of precision and recall
- **Confusion Matrix** — printed to console per model
- **Classification Report** — per-class breakdown printed to console

---

## Visualisations Generated

| File                        | Content                                        |
|-----------------------------|------------------------------------------------|
| `class_distribution.png`    | Bar chart of sample counts per species         |
| `feature_distributions.png` | Histograms of all four features                |
| `pair_plot.png`             | Seaborn pair plot coloured by species          |
| `correlation_heatmap.png`   | Pearson correlation matrix of the four features|
| `box_plots.png`             | Box plots of each feature grouped by species   |
| `model_comparison.png`      | Grouped bar chart comparing all model metrics  |

---

## Saved Model Format

`iris_classifier.pkl` is a Python dictionary serialised with `joblib.dump`:

| Key             | Content                                                                   |
|-----------------|---------------------------------------------------------------------------|
| `model`         | Trained scikit-learn estimator (best model)                               |
| `model_name`    | String name of the best model                                             |
| `scaler`        | Fitted `StandardScaler`                                                   |
| `label_encoder` | Fitted `LabelEncoder`                                                     |
| `feature_names` | `['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']`    |
| `random_state`  | `42`                                                                      |

---

## Code Structure

### `train_model.py` — `IrisClassifier` class

| Method                    | Description                                              |
|---------------------------|----------------------------------------------------------|
| `load_data()`             | Reads `Iris.csv` into a DataFrame                        |
| `explore_data()`          | Prints EDA statistics to stdout                          |
| `visualize_data()`        | Generates and saves the five EDA plots                   |
| `preprocess_data()`       | Cleans, encodes, splits, and scales the data             |
| `initialize_models()`     | Creates the five model instances                         |
| `train_models()`          | Fits all models on the training set                      |
| `evaluate_models()`       | Computes and prints metrics for each model               |
| `compare_models()`        | Prints a comparison table and identifies the best model  |
| `plot_model_comparison()` | Saves the model comparison bar chart                     |
| `save_model()`            | Persists the best model bundle to disk                   |
| `load_model()`            | Static method — loads a saved model bundle from disk     |
| `predict()`               | Returns species prediction, confidence, and probabilities|
| `run_full_pipeline()`     | Runs all steps end-to-end                                |

### `gui.py` — `IrisPredictor` class + Gradio interface

| Component         | Description                                                              |
|-------------------|--------------------------------------------------------------------------|
| `IrisPredictor`   | Loads the `.pkl` file and exposes a `predict()` method with validation   |
| `classify_iris()` | Gradio-facing function — validates inputs, calls predictor, formats output|
| `create_interface()` | Builds and returns the `gr.Blocks` layout                            |
| `main()`          | Launches the Gradio server on `0.0.0.0:7860`                            |

---

## Troubleshooting

**`FileNotFoundError: Dataset not found at: Iris.csv`**  
Ensure `Iris.csv` is in the same directory as `train_model.py` when you run the script, or pass the full path: `IrisClassifier(data_path="/path/to/Iris.csv")`.

**`FileNotFoundError: Model file not found: iris_classifier.pkl`**  
Run `python train_model.py` first to generate the model file before launching `gui.py`.

**`ModuleNotFoundError: No module named 'gradio'`** (or any other package)  
Run `pip install -r requirements.txt` to install all dependencies.

**Plots are saved but not displayed**  
The scripts save plots to disk with `plt.savefig()` rather than calling `plt.show()`. Look for `.png` files in the directory where you ran `train_model.py`.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
