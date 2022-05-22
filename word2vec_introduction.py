import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import warnings
import logging
from gensim.models import word2vec

# warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
warnings.filterwarnings("ignore", message='.*looks like a URL*')
warnings.filterwarnings("ignore", message='.*looks like a filename*')
warnings.filterwarnings("ignore", message='.*looks like a directory name*')
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')


def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review, features="html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    return words


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(
                raw_sentence, remove_stopwords))
    return sentences


if __name__ == '__main__':
    print("Data reading")
    train = pd.read_csv(".\\data\\labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv(".\\data\\unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    print("Data Preprocessing")
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []

    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    print("Model training")
    '''
    num_features = 300
    min_word_count = 40
    num_workers = 4
    context = 10
    downsampling = 1e-3

    print("Training model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers,
                              vector_size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)
    model.init_sims(replace=True)
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    '''
    model = word2vec.Word2Vec.load("./data/300features_40minwords_10context")

    print("Result exploring")
    # model.wv.doesnt_match("man woman child kitchen".split())
    # model.wv.doesnt_match("france england germany berlin".split())
    # model.wv.doesnt_match("france england germany berlin".split())
    # model.wv.doesnt_match("paris berlin london austria".split())
    # model.wv.most_similar("man")
    # model.wv.most_similar("queen")
    # model.wv.most_similar("awful")
