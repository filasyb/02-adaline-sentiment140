import re
import pandas as pd
from src.utils.config_loader import load_config


config = load_config()


def clean_text(text: str) -> str:
    """
    Applies text normalization rules based on the project configuration.
    """
    if config.preprocessing.lowercase:
        text = text.lower()

    if config.preprocessing.remove_urls:
        text = re.sub(r"http\S+|www\S+", "", text)

    if config.preprocessing.remove_mentions:
        text = re.sub(r"@\w+", "", text)

    if config.preprocessing.remove_hashtags:
        text = re.sub(r"#\w+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    Applies text cleaning to a DataFrame and returns a normalized version.
    """
    df = df.copy()
    df[text_column] = df[text_column].fillna("").astype(str)
    df[text_column] = df[text_column].apply(clean_text)
    return df
