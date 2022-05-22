import os
import sys
import logging

from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import pandas as pd
import tensorflow as tf
import numpy as np


train = pd.read_csv("./corpus/imdb/labeledTrainData.tsv", header=0,
    delimiter="\t", quoting=3)
test = pd.read_csv("./corpus/imdb/testData.tsv", header=0,
    delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ''.join(sys.argv))

    train_texts, train_labels, test_texts = [], [], []
    for i, review in enumerate(train["review"]):
        train_texts.append(review)
        train_labels.append(train['sentiment'][i])

    for review in test['review']:
        test_texts.append(review)

    # train_texts, train_labels = train_texts[:100], train_labels[:100]
    # test_texts = test_texts[:100]
    # test_labels = tf.ones(len(test_texts))

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)


    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    ))

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
    ))

    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', finetuning_task='text-classification')
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])  # can also use any keras loss fn
    model.fit(train_dataset.shuffle(1000).batch(8), epochs=3, batch_size=8)

    y_pred = model.predict(test_dataset.batch(8))[0]
    y_pred = np.argmax(y_pred, axis=-1).flatten()

    print(y_pred.shape)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": y_pred})
    result_output.to_csv("./result/bert.csv", index=False, quoting=3)