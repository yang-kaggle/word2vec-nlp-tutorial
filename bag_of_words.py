import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, features="html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stops]
    return " ".join(meaningful_words)


if __name__ == '__main__':
    train = pd.read_csv("./data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    num_reviews = train["review"].size
    clean_train_reviews = []
    for i in range(0, num_reviews):
        clean_train_reviews.append(review_to_words(train["review"][i]))

    test = pd.read_csv("./data/testData.tsv", header=0, delimiter="\t", quoting=3)
    num_reviews = len(test["review"])
    clean_test_reviews = []

    print("cleaning\n")
    for i in range(0, num_reviews):
        clean_test_reviews.append(review_to_words(test["review"][i]))

    vectorizer = CountVectorizer(analyzer="word", tokenizer=None,
                                 preprocessor=None, stop_words=None,
                                 max_features=5000)
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    train_data_features = train_data_features.toarray()
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    print("forest training")
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_data_features, train["sentiment"])

    result = forest.predict(test_data_features)
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output_file = "./result/result_bag_of_words.csv"
    output.to_csv(output_file, index=False, quoting=3)
