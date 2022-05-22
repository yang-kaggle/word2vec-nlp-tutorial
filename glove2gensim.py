from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = 'D:\\glove.840B.300d.txt'
word2vec_output_file = 'D:\\glove.840B.300d.word2vec.txt'
(count, dimensions) = glove2word2vec(glove_input_file, word2vec_output_file)
print(count, '\n', dimensions)