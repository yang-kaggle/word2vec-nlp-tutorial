import os
import sys
import logging
import numpy

from transformers import DistilBertTokenizerFast, TFDistilBertMainLayer, DistilBertConfig
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

    # print(train_labels)

    for review in test['review']:
        test_texts.append(review)

    train_labels = tf.keras.utils.to_categorical(train_labels)
    val_labels = tf.keras.utils.to_categorical(val_labels)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    ))

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
    ))

    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_labels
    ))

    config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
    initializer = tf.keras.initializers.TruncatedNormal(config.initializer_range)
    bert_layer = TFDistilBertMainLayer(config=config)

    input_ids = tf.keras.layers.Input(shape=(config.max_position_embeddings, ), dtype='int32')
    # token_type_ids = tf.keras.Input(shape=(config.max_position_embeddings, ), dtype='int32')
    attention_mask = tf.keras.layers.Input(shape=(config.max_position_embeddings, ), dtype='int32')
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

    bert_output = bert_layer(inputs)
    hidden_states = bert_output[0]

    output = tf.keras.layers.Dense(config.dim, activation='relu', kernel_initializer=initializer)(hidden_states[:, 0])
    output = tf.keras.layers.Dropout(config.dropout)(output)
    output = tf.keras.layers.Dense(config.num_labels, activation='softmax', kernel_initializer=initializer)(output)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])  # can also use any keras loss fn
    model.fit(train_dataset.shuffle(1000).batch(8), validation_data=val_dataset.batch(8), epochs=3, batch_size=8)

    y_pred = model.predict(test_dataset.batch(8))
    y_pred = np.argmax(y_pred, axis=-1).flatten()

    print(y_pred.shape)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": y_pred})
    result_output.to_csv("./result/bert_2.csv", index=False, quoting=3)

