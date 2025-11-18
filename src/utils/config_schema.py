from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Configuration for dataset and feature storage paths."""
    raw_data: str
    preprocessed_data: str
    features_dir: str


class PreprocessingConfig(BaseModel):
    """Preprocessing flags for text normalization."""
    lowercase: bool
    remove_urls: bool
    remove_mentions: bool
    remove_hashtags: bool


class SampleConfig(BaseModel):
    """Sampling configuration for creating balanced subsets."""
    positive_size: int = Field(gt=0)
    negative_size: int = Field(gt=0)
    random_seed: int


class ModelConfig(BaseModel):
    """ADALINE model hyperparameters."""
    learning_rate: float = Field(gt=0)
    epochs: int = Field(gt=0)
    random_seed: int
    threshold: float
    normalize_features: bool


class ProjectConfig(BaseModel):
    """Full project configuration schema."""
    paths: PathsConfig
    preprocessing: PreprocessingConfig
    sample: SampleConfig
    model: ModelConfig
