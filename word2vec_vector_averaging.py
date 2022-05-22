from gensim.models import Word2Vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords


def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review, features="html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.index_to_key)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs


if __name__ == '__main__':
    print("Reading data")
    train = pd.read_csv("./data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("./data/testData.tsv", header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv("./data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    model = Word2Vec.load("./data/300features_40minwords_10context")
    num_features = model.wv.vectors.shape[1]

    print("Creating average vectors")
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model.wv, num_features)

    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model.wv, num_features)

    print("Training a random forest and predicting")
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainDataVecs, train["sentiment"])

    result = forest.predict(testDataVecs)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("./result/result_word2vec_average_vectors.csv", index=False, quoting=3)
