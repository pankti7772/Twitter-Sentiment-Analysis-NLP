# src/components/data_transformation.py
import os
import sys
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Project utilities (fall back gracefully if not present)
try:
    from src.exception import CustomException
except Exception:  # pragma: no cover
    class CustomException(Exception):
        def __init__(self, original_exception, sys_info=None):
            super().__init__(str(original_exception))
            self.original_exception = original_exception
            self.sys_info = sys_info

# Use get_logger if available for consistent logging
try:
    from src.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

try:
    from src.utils import save_object
except Exception:  # pragma: no cover
    def save_object(file_path: str, obj) -> None:
        import joblib
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "tfidf_preprocessor.pkl")
    label_encoder_obj_file_path: str = os.path.join("artifacts", "label_encoder.pkl")


# ---------- Text cleaning transformer ----------
class TextCleaner(TransformerMixin, BaseEstimator):
    """
    Clean raw tweet text:
    - remove urls, mentions, convert hashtags (#tag -> tag)
    - remove extra whitespace, punctuation
    - optional: lemmatization using nltk's WordNetLemmatizer (if installed)
    """
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    MENTION_PATTERN = re.compile(r"@\w+")
    HASHTAG_PATTERN = re.compile(r"#(\w+)")
    RT_PATTERN = re.compile(r"\brt\b", flags=re.IGNORECASE)
    NON_ALPHANUM = re.compile(r"[^A-Za-z0-9\s'’]")  # keep apostrophes for contractions

    def __init__(self, lemmatize: bool = True, lower: bool = True, remove_stopwords: bool = False):
        self.lemmatize = lemmatize
        self.lower = lower
        self.remove_stopwords = remove_stopwords
        # lazy import / setup for lemmatizer and stopwords to avoid heavy imports if not used
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            nltk.data.find('corpora/wordnet')
            self.lemmatizer = WordNetLemmatizer()
        except Exception:
            self.lemmatizer = None
        try:
            from nltk.corpus import stopwords
            self.stopwords = set(stopwords.words("english"))
        except Exception:
            self.stopwords = set()

    def fit(self, X, y=None):
        return self

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        # remove URLs and mentions
        text = self.URL_PATTERN.sub("", text)
        text = self.MENTION_PATTERN.sub("", text)

        # remove standalone RT tokens
        text = self.RT_PATTERN.sub("", text)

        # convert hashtags to plain words (#Hello -> Hello)
        text = self.HASHTAG_PATTERN.sub(r"\1", text)

        # lowercasing
        if self.lower:
            text = text.lower()

        # remove most punctuation/symbols (but keep apostrophes)
        text = self.NON_ALPHANUM.sub(" ", text)

        # collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # optional: lemmatize tokens if WordNet available
        if self.lemmatize and self.lemmatizer:
            tokens = [t for t in text.split() if t]
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            text = " ".join(tokens)

        # optional: remove stopwords
        if self.remove_stopwords and self.stopwords:
            tokens = [t for t in text.split() if t and t not in self.stopwords]
            text = " ".join(tokens)

        return text

    def transform(self, X):
        """
        X can be an array-like of strings or a DataFrame with a 'text' column.
        Returns an array of cleaned strings.
        """
        if isinstance(X, (pd.Series, np.ndarray, list)):
            texts = pd.Series(X)
        elif isinstance(X, pd.DataFrame):
            # attempt to find text column automatically if common names used
            if 'text' in X.columns:
                texts = X['text'].astype(str)
            else:
                # fallback: try common alternatives
                for candidate in ['tweet', 'clean_text', 'tweet_text']:
                    if candidate in X.columns:
                        texts = X[candidate].astype(str)
                        break
                else:
                    raise ValueError("DataFrame input must contain a 'text' (or 'tweet') column")
        else:
            texts = pd.Series(X)

        cleaned = texts.apply(self._clean_text).tolist()
        return np.array(cleaned)


# ---------- Meta feature extractor ----------
class MetaFeatureExtractor(TransformerMixin, BaseEstimator):
    """
    Compute simple numeric features from raw tweet text (input can be a Series or DataFrame with 'text').
    Produces a numpy array of shape (n_samples, n_features).
    Features included:
      - tweet_length (chars)
      - word_count
      - uppercase_ratio
      - exclamation_count
      - question_count
      - hashtag_count
      - mention_count
      - emoji_count (approximate)
    """
    EMOJI_PATTERN = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "]+", flags=re.UNICODE)

    def fit(self, X, y=None):
        return self

    def _features_from_text(self, text: str):
        if not isinstance(text, str):
            text = ""

        tweet_length = len(text)
        word_count = len(text.split())
        uppercase_chars = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_chars / tweet_length if tweet_length > 0 else 0.0
        exclamation_count = text.count("!")
        question_count = text.count("?")
        hashtag_count = text.count("#")
        mention_count = text.count("@")
        # emoji_count (approx)
        emoji_count = len(self.EMOJI_PATTERN.findall(text))
        return [
            tweet_length,
            word_count,
            uppercase_ratio,
            exclamation_count,
            question_count,
            hashtag_count,
            mention_count,
            emoji_count
        ]

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            if 'text' in X.columns:
                texts = X['text'].astype(str).tolist()
            else:
                # fallback to other candidates
                for candidate in ['tweet', 'clean_text', 'tweet_text']:
                    if candidate in X.columns:
                        texts = X[candidate].astype(str).tolist()
                        break
                else:
                    raise ValueError("DataFrame input must contain a 'text' (or 'tweet') column")
        elif isinstance(X, (pd.Series, list, np.ndarray)):
            texts = list(X)
        else:
            texts = list(X)

        features = [self._features_from_text(t) for t in texts]
        return np.array(features, dtype=float)


# ---------- Combined TF-IDF + meta vectorizer ----------
class TextAndMetaVectorizer(TransformerMixin, BaseEstimator):
    """
    Fits a TfidfVectorizer on cleaned text and, on transform, returns the horizontal stack of:
      [tfidf_matrix | meta_numeric_features]
    """
    def __init__(self,
                 tfidf_params: Optional[dict] = None,
                 max_meta_features: int = 8):
        self.tfidf_params = tfidf_params or {
            "ngram_range": (1, 2),
            "min_df": 5,
            "max_df": 0.9,
            "max_features": 30000,
            "analyzer": "word",
            "norm": "l2",
        }
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.text_cleaner = TextCleaner(lemmatize=True, lower=True, remove_stopwords=False)
        self.meta_extractor = MetaFeatureExtractor()
        self.max_meta_features = max_meta_features

    def fit(self, X, y=None):
        """
        X: array-like or DataFrame with text column(s)
        """
        # get cleaned text array
        cleaned = self.text_cleaner.transform(X)
        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        self.vectorizer.fit(cleaned)
        return self

    def transform(self, X):
        cleaned = self.text_cleaner.transform(X)
        tfidf_matrix = self.vectorizer.transform(cleaned)  # sparse matrix

        meta_features = self.meta_extractor.transform(X)  # dense numpy array shape (n_samples, n_meta)
        # Convert meta_features to sparse for efficient hstack if tfidf is sparse
        meta_sparse = sparse.csr_matrix(meta_features)
        combined = sparse.hstack([tfidf_matrix, meta_sparse], format="csr")
        return combined

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


# ---------- DataTransformation orchestrator ----------
class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    @staticmethod
    def _find_text_column(df: pd.DataFrame) -> str:
        """Find a suitable text column name from common candidates."""
        candidates = ['text', 'tweet', 'clean_text', 'tweet_text']
        for c in candidates:
            if c in df.columns:
                return c
        # fallback: if only one column and it's not obviously numeric, use it
        if df.shape[1] == 1:
            return df.columns[0]
        raise ValueError("No text column found. Expected one of: 'text','tweet','clean_text','tweet_text'")

    @staticmethod
    def _find_label_column(df: pd.DataFrame) -> str:
        candidates = ['label', 'sentiment', 'sent']
        for c in candidates:
            if c in df.columns:
                return c
        raise ValueError("No label column found. Expected one of: 'label','sentiment','sent'")

    def initiate_data_transformation(self,
                                     train_path: str,
                                     test_path: str,
                                     convert_to_dense: bool = True
                                     ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Reads train/test CSV files that must include a text column and a label column,
        transforms features (TF-IDF + meta), encodes labels, saves preprocessor + label encoder,
        and returns (train_array, test_array, preprocessor_path) where train/test arrays are
        horizontally stacked [X_features | y_label].
        """
        logger.info("Starting data transformation")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Loaded train and test dataframes")

            # Detect column names
            text_col_train = self._find_text_column(train_df)
            text_col_test = self._find_text_column(test_df)
            label_col_train = self._find_label_column(train_df)
            label_col_test = self._find_label_column(test_df)

            # If column names differ, rename to 'text' and 'label' for consistency
            if text_col_train != 'text':
                train_df = train_df.rename(columns={text_col_train: 'text'})
            if text_col_test != 'text':
                test_df = test_df.rename(columns={text_col_test: 'text'})
            if label_col_train != 'label':
                train_df = train_df.rename(columns={label_col_train: 'label'})
            if label_col_test != 'label':
                test_df = test_df.rename(columns={label_col_test: 'label'})

            # Keep copies
            train_df_original = train_df.copy()
            test_df_original = test_df.copy()

            # Fit transformer on train data
            preprocessor = TextAndMetaVectorizer()
            logger.info("Fitting TF-IDF + meta preprocessor on training data")
            preprocessor.fit(train_df['text'])

            # Transform train and test sets
            X_train_sparse = preprocessor.transform(train_df['text'])
            X_test_sparse = preprocessor.transform(test_df['text'])
            logger.info("Transformed text into feature matrices (TF-IDF + meta)")

            # Encode labels
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(train_df['label'].astype(str))
            y_test = label_encoder.transform(test_df['label'].astype(str))
            logger.info("Encoded labels into numeric form")

            # Convert sparse to dense arrays for downstream components that expect numpy arrays
            if convert_to_dense:
                logger.info("Converting sparse matrices to dense arrays (convert_to_dense=True). "
                            "If your dataset is large, consider setting convert_to_dense=False to keep sparse.")
                X_train = X_train_sparse.toarray()
                X_test = X_test_sparse.toarray()
            else:
                # keep sparse matrices; return them as SciPy CSR arrays with labels separate
                X_train = X_train_sparse
                X_test = X_test_sparse

            # Combine features and labels
            if convert_to_dense:
                train_arr = np.c_[X_train, y_train]
                test_arr = np.c_[X_test, y_test]
            else:
                # For sparse case, stack label as a dense column to the right (still returned as tuple)
                train_arr = (X_train, y_train)
                test_arr = (X_test, y_test)

            # Save the preprocessor and label encoder
            os.makedirs(os.path.dirname(self.config.preprocessor_obj_file_path), exist_ok=True)
            save_object(file_path=self.config.preprocessor_obj_file_path, obj=preprocessor)
            save_object(file_path=self.config.label_encoder_obj_file_path, obj=label_encoder)
            logger.info("Saved preprocessor to %s and label encoder to %s",
                        self.config.preprocessor_obj_file_path, self.config.label_encoder_obj_file_path)

            logger.info("Data transformation completed successfully")
            return train_arr, test_arr, self.config.preprocessor_obj_file_path

        except Exception as e:
            logger.exception("Error in data transformation")
            raise CustomException(e, sys)
