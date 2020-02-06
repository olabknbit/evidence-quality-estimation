import string
from typing import List

import nltk.data
from nltk.corpus import stopwords
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.api import TokenizerI

pubmed_stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
                    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
                    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
                    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
                    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
                    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
                    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
                    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
                    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
                    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
                    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
                    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
                    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
                    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
                    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
                    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]


def remove_stopwords(words: List[str], stopwords: List[str]) -> List[str]:
    words = [w for w in words if w not in stopwords]
    return words


def remove_words_shorter_than(words: List[str], n: int) -> List[str]:
    words = [w for w in words if (len(w) >= n)]
    return words


def remove_numbers(words: List[str]) -> List[str]:
    def check(word: str) -> bool:
        return not any(char.isdigit() for char in word)

    words = list(filter(check, words))
    return words


def remove_punctuation(text: str) -> str:
    no_punc = ''.join([c for c in text if c not in string.punctuation])
    return no_punc


def word_lemmatizer(words: List[str], lemmatizer) -> List[str]:
    words = [lemmatizer.lemmatize(i) for i in words]
    return words


def word_stemmer(words: List[str], stemmer) -> List[str]:
    words = [stemmer.stem(i) for i in words]
    return words


class TextProcessor:
    def __init__(self, sent_tokenizer=nltk.data.load('tokenizers/punkt/PY3/english.pickle'),
                 tokenizer=RegexpTokenizer(r'\w+'), stemmer=PorterStemmer(),
                 remove_numbers=True):
        self.sent_tokenizer: TokenizerI = sent_tokenizer
        self.tokenizer: TokenizerI = tokenizer
        # self.pubmed_stopwords: List[str] = pubmed_stopwords
        self.stemmer: StemmerI = stemmer
        self.remove_numbers: bool = remove_numbers
        self.stopwords = stopwords.words('english') + pubmed_stopwords

    def process_text(self, text: str) -> str:
        sentences = self.sent_tokenizer.tokenize(text)
        sentences = list(map(self.process_sentence, sentences))
        return ' '.join(sentences)

    def process_sentence(self, sentence: str) -> str:
        if sentence == '':
            return ''
        sentence = remove_punctuation(sentence.lower())
        words = self.tokenizer.tokenize(sentence)
        words = remove_stopwords(words, self.stopwords)
        # words = word_lemmatizer(words, self.lemmatizer)
        words = word_stemmer(words, self.stemmer)

        words = remove_words_shorter_than(words, 3)
        if self.remove_numbers:
            words = remove_numbers(words)
        return ' '.join(words)


class TextTokenizer:
    def __init__(self):
        self.tp = TextProcessor()

    def preprocess(self, text):
        # TODO maybe make it more clever
        if text == '':
            text = "empty"
        res = self.tp.process_text(text).split(' ')
        return res


class PublicationYearTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X: List[str], y=None):
        import numpy as np

        def f(X) -> List[List[float]]:
            new_X = []
            for xs in X:
                new_xs = [float(x) for x in xs]
                x = [float(np.median(new_xs))]
                new_X.append(x)
            return new_X

        X = [ref_ids.split(' ') for ref_ids in X]
        X = f(X)
        return np.array(X)
