import pandas as pd
from src.utils.config_loader import load_config
from src.preprocessing.data_preprocessing import preprocess_dataframe


config = load_config()


def generate_sample() -> pd.DataFrame:
    """
    Loads the Sentiment140 dataset, preprocesses it, creates a balanced sample,
    and saves it to the preprocessed data path.
    """
    df = pd.read_csv(
        config.paths.raw_data,
        encoding="latin-1",
        header=None,
        names=["sentiment", "id", "date", "query", "user", "text"]
    )

    df = preprocess_dataframe(df, text_column="text")

    df_pos = df[df["sentiment"] == 4].sample(
        n=config.sample.positive_size,
        random_state=config.sample.random_seed
    )

    df_neg = df[df["sentiment"] == 0].sample(
        n=config.sample.negative_size,
        random_state=config.sample.random_seed
    )

    sample_df = pd.concat([df_pos, df_neg]).sample(
        frac=1,
        random_state=config.sample.random_seed
    )

    sample_df.to_csv(
        config.paths.preprocessed_data,
        index=False,
        encoding="utf-8"
    )

    return sample_df


if __name__ == "__main__":
    df = generate_sample()
    print(df.head())
