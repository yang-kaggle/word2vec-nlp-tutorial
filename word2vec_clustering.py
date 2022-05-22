from gensim.models import Word2Vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.cluster import KMeans


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
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


def create_bag_of_centroids(wordlist, word_centroid_map):
    num_centroids = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids


if __name__ == '__main__':
    print("data reading")
    train = pd.read_csv("./data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("./data/testData.tsv", header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv("./data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    model = Word2Vec.load("./data/300features_40minwords_10context")

    print("data cleaning")
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

    print("K-means clustering")
    word_vectors = model.wv.vectors
    num_clusters = int(word_vectors.shape[0] / 5)
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)
    word_centroid_map = dict(zip(model.wv.index_to_key, idx))

    print("Creating bag of centroids")
    train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")
    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    print("forest training")
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_centroids, train["sentiment"])
    result = forest.predict(test_centroids)

    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("./result/result_word2vec_clustering.csv", index=False, quoting=3)
