
import os

import re

import unicodedata
from num2words import num2words
from nltk import word_tokenize


import nltk


nltk.data.path.append(os.path.abspath('./nltk_data'))
nltk.download('punkt', quiet=True, raise_on_error=True,
              download_dir='./nltk_data')
nltk.download('stopwords', quiet=True, raise_on_error=True,
              download_dir='./nltk_data')
nltk.download('wordnet', quiet=True, raise_on_error=True,
              download_dir='./nltk_data')
ROOT_DIR = os.path.abspath(os.curdir)
NLTK_DATA_DIR = os.path.join(ROOT_DIR, 'nltk_data')
nltk.data.path.append(NLTK_DATA_DIR)


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    return [x.lower() for x in words]


def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""

    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', ' ', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = num2words(word, lang='es_CO')
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    stopword_es = nltk.corpus.stopwords.words('spanish')
    new_words = []

    for word in words:
        if word not in stopword_es:
            new_words.append(word)
    return new_words


def preprocessing(words):
    words = to_lowercase(words)
    words = replace_numbers(words)
    words = remove_punctuation(words)
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words


def clean_text(reviews):
    print(reviews)
    reviews = reviews.apply(word_tokenize).apply(preprocessing)

    reviews = reviews.apply(lambda x: ' '.join(map(str, x)))

    reviews = reviews.apply(word_tokenize).apply(preprocessing)
    reviews = reviews.apply(lambda x: ' '.join(map(str, x)))

    return reviews
