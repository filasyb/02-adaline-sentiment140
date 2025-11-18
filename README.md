# ADALINE Sentiment Classifier — Machine Learning From Scratch

This repository implements an **ADALINE (Adaptive Linear Neuron)** classifier from scratch using Python and NumPy.  
The model is trained on the **Sentiment140** dataset, which contains 1.6M tweets labeled as positive or negative.

The purpose of this project is to build a fully manual, interpretable machine learning workflow and demonstrate clear understanding of:

- ML fundamentals
- Clean architecture
- Preprocessing pipelines
- Reproducible configuration using Pydantic
- Debug-ready scripts
- Modular code for training, inference, and data preparation

---

## Project Objectives

- Implement ADALINE using gradient descent (no ML frameworks).
- Build a modular, production-style ML architecture.
- Apply text preprocessing and vectorization.
- Train and evaluate a linear classifier on real-world sentiment data.
- Provide a reusable structure for future models.

---

## Project Structure

```

project-root/
│
├── config/
│   └── local.yaml
│
├── data/
│   ├── 01_raw/
│   ├── 02_preprocessed/
│   └── 03_models/
│
├── notebooks/
│   └── 01_adaline_sentiment140_exploration.ipynb
│
├── src/
│   ├── data/
│   │   └── generate_sample.py
│   ├── models/
│   │   └── adaline.py
│   ├── predict/
│   │   └── predict_adaline.py
│   ├── preprocessing/
│   │   └── data_preprocessing.py
│   ├── train/
│   │   └── train_adaline.py
│   └── utils/
│       ├── config_loader.py
│       └── config_schema.py
│
├── .gitignore
├── environment.yml
└── README.md

````

---

## Installation

```bash
conda env create -f environment.yml
conda activate adaline-env
````

---

## Configuration (Pydantic + YAML)

Configuration is defined in:

```
config/local.yaml
```

Example:

```yaml
paths:
  raw_data: "data/01_raw/training.1600000.processed.noemoticon.csv"
  preprocessed_data: "data/02_preprocessed/sentiment140_sample.csv"

sample:
  positive_size: 5000
  negative_size: 5000
  random_seed: 42

model:
  learning_rate: 0.0001
  epochs: 30
  threshold: 0.0
  normalize_features: true
  random_seed: 42
```

---

## Text Preprocessing

The preprocessing pipeline includes:

* Lowercasing
* URL removal
* Username and hashtag removal
* Special character filtering
* Whitespace normalization

All behavior is controlled by configuration flags in `local.yaml`.

---

## Dataset Sampling

Generate a clean, balanced dataset:

```bash
python src/data/generate_sample.py
```

This script:

1. Loads the 1.6M raw Sentiment140 tweets
2. Cleans and normalizes text
3. Extracts a 10,000-tweet balanced subset
4. Saves it in `data/02_preprocessed/`

---

## Model Training (ADALINE)

Train the classifier:

```bash
python src/train/train_adaline.py
```

The training script performs:

* Vectorization using Bag-of-Words
* Optional feature normalization
* Gradient descent optimization
* Loss history tracking
* Accuracy and classification report generation

Example output:

```
Accuracy: 0.48
Loss (first 5 epochs): [1.91, 1.91, 1.90, ...]
```

> As expected, ADALINE is a very simple linear model and serves as a baseline.

---

## Inference

Run prediction script:

```bash
python src/predict/predict_adaline.py
```

In Python:

```python
from src.predict.predict_adaline import predict_text
result = predict_text("I love this product!", model, vectorizer, scaler)
print(result)
```

Output:

```json
{
  "clean_text": "i love this product",
  "score": 0.42,
  "prediction": 1
}
```

---

## Notebook Included

The notebook:

```
notebooks/01_adaline_sentiment140_exploration.ipynb
```

Contains:

* Exploratory data analysis
* Length distributions
* Class distribution visualization
* Training diagnostics
* Loss curve
* Confusion matrix
* Manual predictions

---

## Best Practices Used

* Clean modular architecture (inspired by ML production templates)
* Pydantic-validated configuration
* Separation of concerns (data → preprocessing → model → inference)
* Debug-ready scripts using `if __name__ == "__main__":`
* Conda environment for full reproducibility
* Version-controlled `.gitignore` optimized for ML

---

## Future Improvements

* Add TF-IDF vectorization
* Introduce PCA for dimensionality reduction
* Implement ADALINE with SGD
* Add logistic regression and SVM baselines
* Introduce experiment tracking (MLflow)
* Deploy inference API using FastAPI or Cloud Run

---

## Author

**Juan David Jaramillo**
Data Scientist · ML Engineer
Focused on: ML systems, NLP, fraud detection, AI for contact centers, GCP & Generative AI.

---