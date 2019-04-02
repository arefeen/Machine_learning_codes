import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

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

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size = 0.25, random_state = 1000) # patition the data into training and testing

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train) # constuct the dictionary of words from the training sentences

X_train = vectorizer.transform(sentences_train) # construct the feature vectors for the sentences in training data using the dictionary
X_test = vectorizer.transform(sentences_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

score = classifier.score(X_test, y_test)

print "Accuracy: " + str(score) # Shows the performance of the model trained on yelp data

for source in df['source'].unique():
	df_source = df[df['source'] == source]
	sentences = df_source['sentence'].values
	y = df_source['label'].values

	sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size = 0.25, random_state = 1000)

	vectorizer = CountVectorizer()
	vectorizer.fit(sentences_train)
	X_train = vectorizer.transform(sentences_train)
	X_test = vectorizer.transform(sentences_test)

	classifier = LogisticRegression()
	classifier.fit(X_train, y_train)
	score = classifier.score(X_test, y_test)

	print 'Accuracy for {} data: {:.4f}'.format(source, score)
