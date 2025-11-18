import pandas as pd
from src.preprocessing.data_preprocessing import preprocess_dataframe


def predict_text(text: str, model, vectorizer, scaler=None):
    """
    Predicts the sentiment of a given text using a trained ADALINE model.
    Returns the cleaned text, prediction score, and binary class.
    """
    df = pd.DataFrame({"text": [text]})
    df_clean = preprocess_dataframe(df, text_column="text")

    X = vectorizer.transform(df_clean["text"]).toarray()

    if scaler is not None:
        X = scaler.transform(X)

    scores = model.predict_scores(X)
    preds = model.predict(X)

    return {
        "clean_text": df_clean["text"].iloc[0],
        "score": float(scores[0]),
        "prediction": int(preds[0])
    }


if __name__ == "__main__":
    from src.train.train_adaline import train_adaline

    result = train_adaline()
    model = result["model"]
    vectorizer = result["vectorizer"]
    scaler = result["scaler"]

    sample = "I love this so much!"
    print(predict_text(sample, model, vectorizer, scaler))
