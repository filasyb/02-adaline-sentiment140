import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from src.utils.config_loader import load_config
from src.preprocessing.data_preprocessing import preprocess_dataframe
from src.models.adaline import Adaline


def train_adaline():
    """
    Trains the ADALINE model using the preprocessed Sentiment140 dataset.
    Returns the trained model, vectorizer, scaler, and metrics.
    """
    config = load_config()

    df = pd.read_csv(config.paths.preprocessed_data)
    df = preprocess_dataframe(df, text_column="text")

    X_text = df["text"].astype(str).fillna("")
    y = df["sentiment"].replace({4: 1, 0: 0}).values

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_text).toarray()

    scaler = None
    if config.model.normalize_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=config.model.random_seed,
        stratify=y
    )

    model = Adaline(
        learning_rate=config.model.learning_rate,
        epochs=config.model.epochs,
        random_state=config.model.random_seed,
        threshold=config.model.threshold,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "loss_history": model.loss_history_,
    }

    return {
        "model": model,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "metrics": metrics,
    }


if __name__ == "__main__":
    result = train_adaline()
    print("Accuracy:", result["metrics"]["accuracy"])
    print("First 5 loss values:", result["metrics"]["loss_history"][:5])
