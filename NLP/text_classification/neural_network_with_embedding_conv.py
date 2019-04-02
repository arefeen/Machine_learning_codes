from keras_tokenizer_example import data_initialization, get_train_and_test_data, create_tokenizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras .models import Sequential
from keras import layers

def create_model(vocab_size, embedding_dim, maxleng):
	model = Sequential()
	model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxleng))
	model.add(layers.Conv1D(128, 5, activation='relu'))
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
	embedding_dim = 100

	tokenizer = create_tokenizer(sentences_train)
	X_train = tokenizer.texts_to_sequences(sentences_train)
	X_test = tokenizer.texts_to_sequences(sentences_test)

	X_train = pad_sequences(X_train, padding = 'post', maxlen = maxleng)
	X_test = pad_sequences(X_test, padding = 'post', maxlen = maxleng)

	model = create_model(vocab_size, embedding_dim, maxleng)
	history = model.fit(X_train, y_train, epochs=40, verbose=False, validation_data=(X_test, y_test), batch_size=10)
	
	loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
	print "Training Accuracy: {:.4f}".format(accuracy)
	loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
	print "Testing Accuracy:  {:.4f}".format(accuracy)

if __name__ == "__main__":
	main()
