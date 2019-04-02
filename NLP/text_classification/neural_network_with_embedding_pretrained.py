from keras_tokenizer_example import data_initialization, get_train_and_test_data, create_tokenizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras .models import Sequential
from keras import layers
import numpy as np

def create_embedding_matrix(filepath, word_index, embedding_dim):
    	vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
	embedding_matrix = np.zeros((vocab_size, embedding_dim))

	with open(filepath) as f:
		for line in f:
			element = line.split()
			word = element[0]
			vector = element[1:]
			if word in word_index:
				idx = word_index[word] 
				embedding_matrix[idx] = np.array(vector, dtype = np.float32)[:embedding_dim]

	return embedding_matrix


def create_model(vocab_size, embedding_dim, maxleng, embedding_matrix):
	model = Sequential()
	model.add(layers.Embedding(vocab_size, embedding_dim, weights = [embedding_matrix], input_length = maxleng, trainable = True))
	model.add(layers.GlobalMaxPool1D())
	#model.add(layers.Flatten())
	model.add(layers.Dense(10, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	print model.summary()
	return model

def main():
	sentences, y = data_initialization()
	sentences_train, sentences_test, y_train, y_test = get_train_and_test_data(sentences, y)

	tokenizer = create_tokenizer(sentences_train)
	
	vocab_size = len(tokenizer.word_index) + 1
	maxleng = 100
	embedding_dim = 50

	tokenizer = create_tokenizer(sentences_train)
	X_train = tokenizer.texts_to_sequences(sentences_train)
	X_test = tokenizer.texts_to_sequences(sentences_test)

	X_train = pad_sequences(X_train, padding = 'post', maxlen = maxleng)
	X_test = pad_sequences(X_test, padding = 'post', maxlen = maxleng)

	embedding_matrix = create_embedding_matrix('../../data/precomputed_embedding_space/glove.6B.50d.txt', tokenizer.word_index, embedding_dim)

	model = create_model(vocab_size, embedding_dim, maxleng, embedding_matrix)
	history = model.fit(X_train, y_train, epochs=50, verbose=False, validation_data=(X_test, y_test), batch_size=10)
	
	loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
	print "Training Accuracy: {:.4f}".format(accuracy)
	loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
	print "Testing Accuracy:  {:.4f}".format(accuracy)

if __name__ == "__main__":
	main()
