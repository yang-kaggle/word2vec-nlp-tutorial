import os
import sys
import logging

from transformers import BertTokenizerFast, TFBertForSequenceClassification
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

    train_ratio = 0.9
    train_texts, train_labels, val_texts, val_labels, test_texts = [], [], [], [], []
    for i, review in enumerate(train["review"]):
        if np.random.rand() < train_ratio:
            train_texts.append(review)
            train_labels.append(train['sentiment'][i])
        else:
            val_texts.append(review)
            val_labels.append(train['sentiment'][i])

    for review in test['review']:
        test_texts.append(review)

    # train_texts, train_labels = train_texts[:100], train_labels[:100]
    # test_texts = test_texts[:100]
    # test_labels = tf.ones(len(test_texts))

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    ))

    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_labels
    ))

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
    ))

    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', finetuning_task='text-classification',
                                                            num_labels=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6, epsilon=1e-8)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # can also use any keras loss fn
    model.fit(train_dataset.shuffle(1000).batch(4), validation_data=val_dataset.batch(8), epochs=3, batch_size=8)

    y_pred = model.predict(test_dataset.batch(4))[0]
    y_pred = np.argmax(y_pred, axis=-1).flatten()

    print(y_pred.shape)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": y_pred})
    result_output.to_csv("./result/bert_predefine.csv", index=False, quoting=3)
