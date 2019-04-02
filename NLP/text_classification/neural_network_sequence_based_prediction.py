from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras .models import Sequential
from keras import layers
from keras.utils import np_utils
from array import array
import numpy as np


def create_model(vocab_size, embedding_dim, maxleng):
	model = Sequential()
	model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxleng-1))
	model.add(layers.LSTM(50))
	model.add(layers.Dense(vocab_size, activation='softmax'))
	print(model.summary())
	
	return model
	
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text

def main():
	data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts([data])
	vocab_size = len(tokenizer.word_index) + 1

	sequences = list()
	for line in data.split('\n'):
		encoded = tokenizer.texts_to_sequences([line])[0]
		for i in range(1, len(encoded)):
			sequence = encoded[:i+1]
			sequences.append(sequence)
	print('Total Sequences: %d' % len(sequences))

	max_length = max([len(seq) for seq in sequences])
	sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
	print('Max Sequence Length: %d' % max_length)

	sequences = np.array(sequences)
	X, y = sequences[:,:-1],sequences[:,-1]
	y = np_utils.to_categorical(y, num_classes=vocab_size)

	model = create_model(vocab_size, 10, max_length)

	# compile network
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# fit network
	model.fit(X, y, epochs=500, verbose=2)
	
	print(generate_seq(model, tokenizer, max_length-1, 'Jack', 4))
	print(generate_seq(model, tokenizer, max_length-1, 'Jill', 4))



if __name__ == "__main__":
	main()
