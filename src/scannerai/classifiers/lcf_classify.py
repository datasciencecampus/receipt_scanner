"""classify input text to COICOP CODE using trained models."""

import os
import pickle

import joblib
import numpy as np
import pandas as pd


class lcf_classifier:
    """a classifier class for initialization and prediction."""

    def __init__(self, trainedmodel, labelcoder):
        # Initializes the classifier with a model and vectorizer.
        # :param trainedmodel: A trained classifier model that supports predict and predict_proba methods
        # :param labelcoder: A label coder file generated during the training stage

        self.classifier = None
        self.labelCoder = None
        self.InitSuccess = False  # Initialize InitSuccess to False

        if (
            trainedmodel is None
            or not os.path.exists(trainedmodel)
            or not os.path.isfile(trainedmodel)
        ):
            print(f"WARNING: the trained model does not exist!")
        else:
            self.classifier = joblib.load(trainedmodel)
            print(f"trained model {trainedmodel} loaded successfully!")

        if (
            labelcoder is None
            or not os.path.exists(labelcoder)
            or not os.path.isfile(labelcoder)
        ):
            print(f"WARNING: label coder does not exist!")
        else:
            with open(labelcoder, "rb") as f:
                self.labelCoder = pickle.load(f)

        # Set InitSuccess to True only if both classifier and labelCoder are loaded
        if self.classifier is not None and self.labelCoder is not None:
            self.InitSuccess = True

    def get_InitSuccess(self):
        """Return the initialization status."""
        return self.InitSuccess

    def predict_dataframe(self, feature_df):
        """
        Predicts class labels and probabilities for a given DataFrame of features.

        :param feature_df: DataFrame containing the features for classification
        :return: DataFrame with predicted labels and probabilities
        """
        if not self.get_InitSuccess():
            return None

        print("Processing DataFrame input:\n", feature_df)

        # Get predictions
        predictions = self.classifier.predict(feature_df)
        prediction_decode = self.labelCoder.inverse_transform(predictions)

        probabilities = self.classifier.predict_proba(feature_df)
        maxscores = (
            (
                probabilities[
                    np.arange(len(probabilities)),
                    np.argmax(probabilities, axis=1),
                ]
            )
            * 100
        )
        maxscores = np.round(maxscores, 2)

        # Create a results DataFrame
        results_df = pd.DataFrame(
            {"predictions": prediction_decode, "probabilities": maxscores}
        )

        return results_df

    def predict_string(self, item_desc):
        """
        Predicts class label and probability for a single text description.

        :param item_desc: String containing the text description for classification
        :return: tuple (predicted_label, probability)
        """

        if not self.get_InitSuccess():
            return None, None

        # print("Processing string input:", item_desc)

        # Clean and prepare input
        item_desc = str(item_desc).strip()

        # Convert single string to array-like format
        item_desc_array = np.array([item_desc])

        # Get prediction
        prediction = self.classifier.predict(item_desc_array)
        prediction_decode = self.labelCoder.inverse_transform(prediction)

        # Get probability
        probabilities = self.classifier.predict_proba(item_desc_array)
        maxscore = (probabilities[0, np.argmax(probabilities, axis=1)]) * 100
        maxscore = np.round(maxscore, 2)

        return prediction_decode[0], float(maxscore)

    def predict(self, input_data):
        """
        Universal predict method that handles both DataFrame and string inputs.

        :param input_data: Either a DataFrame or a string
        :return: DataFrame with results for DataFrame input, or tuple (prediction, probability) for string input
        """
        if isinstance(input_data, pd.DataFrame):
            return self.predict_dataframe(input_data)
        elif isinstance(input_data, str):
            return self.predict_string(input_data)
        elif isinstance(input_data, pd.Series):
            return self.predict_dataframe(input_data.to_frame())
        else:
            raise ValueError(
                "Input must be either a DataFrame, Series, or string"
            )
