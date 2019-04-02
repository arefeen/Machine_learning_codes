from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras .models import Sequential
from keras import layers
from keras.utils import np_utils
from array import array
import numpy as np


def create_model(vocab_size, embedding_dim, maxleng):
	model = Sequential()
	model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxleng))
	model.add(layers.LSTM(50))
	model.add(layers.Dense(vocab_size, activation='softmax'))
	print(model.summary())

	return model

def generate_seq(model, tokenizer, seed_text, n_words):
	in_text, result = seed_text, seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		encoded = np.array(encoded)
		# predict a word in the vocabulary
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text, result = out_word, result + ' ' + out_word
	return result

def main():
	data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts([data])
	encoded = tokenizer.texts_to_sequences([data])[0]

	vocab_size = len(tokenizer.word_index) + 1
	print 'Vocabulary size {:.4f}'.format(vocab_size)

	sequences = []
	X = []
	y = []
	for i in range(1, len(encoded)):
		sequence = encoded[i-1:i+1]
		sequences.append(sequence)
		X.append(sequence[0])
		y.append(sequence[1])
	print('Total Sequences: %d' % len(sequences))

	X = np.array(X)

	y = np_utils.to_categorical(y, num_classes=vocab_size)	
	model = create_model(vocab_size, 10, 1)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(X, y, epochs=500, verbose=2)

	print generate_seq(model, tokenizer, 'Jack', 10)


if __name__ == "__main__":
	main()
