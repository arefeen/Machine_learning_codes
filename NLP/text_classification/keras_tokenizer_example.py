import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def data_initialization():
	filepath_dict = {'yelp':'/bigdata/jianglab/aaref001/Machine_Learning/data/sentiment_labelled_data/yelp_labelled.txt',
			'amazon':'/bigdata/jianglab/aaref001/Machine_Learning/data/sentiment_labelled_data/amazon_cells_labelled.txt',
			'imdb':'/bigdata/jianglab/aaref001/Machine_Learning/data/sentiment_labelled_data/imdb_labelled.txt'}

	df_list = []
	for source, filepath in filepath_dict.items():
		df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
		df['source'] = source  # Add another column filled with the source name
		df_list.append(df)

	df = pd.concat(df_list)
	#print(df.iloc[0])

	df_yelp = df[df['source'] == 'yelp'] # taking only the yelp data
	sentences = df_yelp['sentence'].values # this gives the sentences in numpy array
	y = df_yelp['label'].values # this gives the label in numpy array

	return sentences, y

def get_train_and_test_data(sentences, y):
	sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size = 0.25, random_state = 1000) # patition the data into training and testing

	return sentences_train, sentences_test, y_train, y_test

def create_tokenizer(sentences_train, num_words = 5000):
	tokenizer = Tokenizer(num_words)
	tokenizer.fit_on_texts(sentences_train)

	return tokenizer


def main():
	sentences, y = data_initialization()
	sentences_train, sentences_test, y_train, y_test = get_train_and_test_data(sentences, y)

	tokenizer = create_tokenizer(sentences_train)
	X_train = tokenizer.texts_to_sequences(sentences_train)
	X_test = tokenizer.texts_to_sequences(sentences_test)

	vocab_size = len(tokenizer.word_index) + 1

	print 'Vocabulary size {}'.format(vocab_size)
	
	print sentences_train[2]
	print X_train[2]

	maxlen = 100

	X_train = pad_sequences(X_train, padding = 'post', maxlen = maxlen)
	X_test = pad_sequences(X_test, padding = 'post', maxlen = maxlen)

	print X_train[2]

if __name__ == "__main__":
	main()
