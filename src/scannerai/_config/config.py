"""Class to hold config parameters."""

import os
from io import StringIO

from dotenv import load_dotenv


def load_config(config_file):
    """Load configuration from a text file."""
    try:
        with open(config_file, "r") as f:
            config_content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    # Use StringIO to create a file-like object from the config content
    config_stream = StringIO(config_content)
    load_dotenv(stream=config_stream)

    config = {
        # Debug and Processing Settings
        "DEBUG_MODE": os.getenv("DEBUG_MODE", "False").lower() == "true",
        "ENABLE_PREPROCESSING": os.getenv(
            "ENABLE_PREPROCESSING", "False"
        ).lower()
        == "true",
        "SAVE_PROCESSED_IMAGE": os.getenv(
            "SAVE_PROCESSED_IMAGE", "False"
        ).lower()
        == "true",
        "ENABLE_PRICE_COUNT": os.getenv("ENABLE_PRICE_COUNT", "False").lower()
        == "true",
        # OCR Model Settings
        "OCR_MODEL": int(os.getenv("OCR_MODEL", "3")),  # Default to Gemini
        # Classification Model Paths
        "CLASSIFIER_MODEL_PATH": os.getenv("CLASSIFIER_MODEL_PATH"),
        "LABEL_ENCODER_PATH": os.getenv("LABEL_ENCODER_PATH"),
        # API Keys
        "GEMINI_API_KEY_PATH": os.getenv("GEMINI_API_KEY_PATH"),
        "OPENAI_API_KEY_PATH": os.getenv("OPENAI_API_KEY_PATH"),
        # Google Cloud credentials
        "GOOGLE_CREDENTIALS_PATH": os.getenv("GOOGLE_CREDENTIALS_PATH"),
    }

    # Validate required configurations
    # required_configs = ['CLASSIFIER_MODEL_PATH', 'LABEL_ENCODER_PATH']
    # missing_configs = [key for key in required_configs if not config[key]]

    # if missing_configs:
    #     raise ValueError(f"Missing required configuration(s): {', '.join(missing_configs)}")

    return config


# Create a Config class to handle configuration
class Config:
    """Class to handle configuration."""

    _instance = None
    _config = None

    def __new__(cls, config_file):
        """Create a configuration instance with config_file."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._config = load_config(config_file)
        return cls._instance

    @property
    def debug_mode(self):
        """Get DEBUG_MODE."""
        return self._config["DEBUG_MODE"]

    @property
    def enable_preprocessing(self):
        """Get ENABLE_PREPROCESSING."""
        return self._config["ENABLE_PREPROCESSING"]

    @property
    def save_processed_image(self):
        """Get SAVE_PROCESSED_IMAGE."""
        return self._config["SAVE_PROCESSED_IMAGE"]

    @property
    def enable_price_count(self):
        """Get ENABLE_PRICE_COUNT."""
        return self._config["ENABLE_PRICE_COUNT"]

    @property
    def ocr_model(self):
        """Get OCR_MODEL."""
        return self._config["OCR_MODEL"]

    @property
    def classifier_model_path(self):
        """Get CLASSIFIER_MODEL_PATH."""
        return self._config["CLASSIFIER_MODEL_PATH"]

    @property
    def label_encoder_path(self):
        """Get LABEL_ENCODER_PATH."""
        return self._config["LABEL_ENCODER_PATH"]

    @property
    def gemini_api_key_path(self):
        """Get GEMINI_API_KEY_PATH."""

        return self._config["GEMINI_API_KEY_PATH"]

    @property
    def openai_api_key_path(self):
        """Get OPENAI_API_KEY_PATH."""
        return self._config["OPENAI_API_KEY_PATH"]

    @property
    def google_credentials_path(self):
        """Get GOOGLE_CREDENTIALS_PATH."""
        return self._config["GOOGLE_CREDENTIALS_PATH"]


# Create a global instance
config = Config("config.txt")

# Usage example:
# from scannerai.config.config import config
# if config.debug_mode:
#     print("Debug mode is enabled")
