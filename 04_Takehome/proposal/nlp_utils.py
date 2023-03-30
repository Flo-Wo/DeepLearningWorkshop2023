# Copyright (C) 2022 Florian Wolf
#
# This program is licensed under the Creative Commons Attribution 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
import re
from utils.decorators import log_time
from collections import Counter
from nltk.corpus import stopwords
import numpy as np
from typing import Tuple, Union


def preprocess_str(word: str) -> str:
    """Preprocess strings of our dataset, this includes:
    Remove:
        - all non-word characters (except for numbers and letters)
        - replace multiple whitespaces with no spaces
        - replace all digits by with no space

    Parameters
    ----------
    word : str
        Input string to be preprocessed.

    Returns
    -------
    str
        Preprocessed string.
    """
    # TODO: 1,a)
    # =================
    # Remove all non-word characters (except numbers and letters)
    word = re.sub(r"[^\w\s]", "", word)
    # Replace all runs of whitespaces with no space
    word = re.sub(r"\s+", "", word)
    # replace digits with no space
    word = re.sub(r"\d", "", word)
    # =================
    return word


@log_time
def _most_common_words(
    input_data_raw: Union[list[str], np.ndarray], top_n_words: int = 1000
) -> list[str]:
    """Determine the most common words in the input data and return
    the top_n_words based on their frequency.

    Parameters
    ----------
    input_data_raw : Union[list[str], np.ndarray]
        Input data, list of strings or np.ndarray of strings
        containing the reviews.
    top_n_words : int, optional
        Number of words the vocabulary should be based on,
        by default 1000.

    Returns
    -------
    list[str]
        List of strings with the top_n_words in the vocabulary.
    """
    # TODO: 1,b)
    # =================
    word_list = []
    # define stop word, i.e. where we want to split the tokens
    stop_words = set(stopwords.words("english"))

    for review in input_data_raw:
        for word in review.lower().split():
            word_processed = preprocess_str(word)
            if word_processed not in stop_words and word_processed != "":
                word_list.append(word_processed)
    # now the define the word vocabulary with its frequencies, i.e. we want to count
    # hashable objects
    # see: https://docs.python.org/3/library/collections.html#collections.Counter
    vocabulary = Counter(word_list)

    # extract the n most common words
    most_common_words = [
        word_count[0] for word_count in vocabulary.most_common(top_n_words)
    ]
    # =================
    return most_common_words


@log_time
def tokenize(
    input_data_raw: np.ndarray,
    labels: np.ndarray,
    top_n_words: int = 1000,
) -> Tuple[list[list[int]], np.ndarray, dict]:
    """Tokenize all reviews, i.e. compute the most common words, preprocess all strings
    and only save words which are included in the vocabulary. Transform the labels from
    string format to float.

    Parameters
    ----------
    input_data_raw : np.ndarray
        Input data of the .csv-file.
    labels : np.ndarray
        Labels provided by the .csv-file.
    top_n_words : int, optional
        Number of words our vocabulary should include, by default 1000.

    Returns
    -------
    Tuple[list[list[int]], np.ndarray, dict]
        Cleaned input data, labels as floats,
        mapping between words and their corresponding index
    """
    # TODO: 1,c)
    # =================
    # get the most common words
    most_common_words = _most_common_words(input_data_raw, top_n_words=top_n_words)

    # now we need to define a one-hot vector encoding
    onehot_dict = {word: idx + 1 for idx, word in enumerate(most_common_words)}

    input_data_clean = []
    for review in input_data_raw:
        input_data_clean.append(
            [
                onehot_dict[preprocess_str(word)]
                for word in review.lower().split()
                if preprocess_str(word) in onehot_dict.keys()
            ]
        )

    label_int = [1 if label == "positive" else 0 for label in labels]
    # =================
    # NOTE: we need the labels to be float32 in order to be converted to torch.long afterwards
    return input_data_clean, np.array(label_int, dtype=np.float32), onehot_dict


@log_time
def add_padding(sentences: list[list[int]], target_length: int) -> np.ndarray:
    """Add an (optional) padding to all reviews, i.e. all reviews should have the
    same size to enable batch training. Longer reviews need to be shortened and
    shorter reviews are padded with zeros.

    Parameters
    ----------
    sentences : list[list[int]]
        Tokenized sentences, each review is represented by a list and should
        be converted to a column.
    target_length : int
        Size which all of the reviews should have. Analyze the distribtion
        of length to adjust this parameter properly.

    Returns
    -------
    np.ndarray
        Padded reviews.
    """
    # TODO: 1,d)
    # =================
    features = np.zeros((len(sentences), target_length), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            # either fill with zeros (shorter sentences)
            # or cut of the end (longer sentences)
            features[ii, -len(review) :] = np.array(review)[:target_length]
    # =================
    return features


@log_time
def prepare_data(
    input_data_raw: np.ndarray,
    labels_raw: np.ndarray,
    top_n_words: int = 1000,
    target_length: int = 500,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Complete preprocessing pipeline:
        - Preprocess all strings
        - Tokenize the reviews
        - Add padding

    Parameters
    ----------
    input_data_raw : np.ndarray
        Reviews of the raw .csv-table.
    labels_raw : np.ndarray
        Labels of the raw .csv-table.
    top_n_words : int, optional
        Length of the vocabulary, by default 1000.
    target_length : int, optional
        Maximum length of each review, by default 500.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, dict]
        Cleaned input data, float-type labels and the vocabulary index.
    """
    # TODO: 1,e)
    # =================
    input_data_clean, labels, vocabulary = tokenize(
        input_data_raw,
        labels_raw,
        top_n_words=top_n_words,
    )
    input_data = add_padding(input_data_clean, target_length=target_length)
    # =================
    return input_data, labels, vocabulary
