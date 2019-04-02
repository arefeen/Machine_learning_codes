from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

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

input_dim = X_train.shape[1] # Number of features

input_data = Input(shape=(input_dim,), sparse=True)
den1 = Dense(10, input_shape = (input_dim,), activation = 'relu')(input_data)
den2 = Dense(1, activation = 'sigmoid')(den1)
model = Model(inputs = [input_data], outputs = den2)
#model = Sequential()
#model.add(Dense(10, input_dim = input_dim, activation = 'relu'))
#model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#model.summary()

#print X_train.type
#print y_train.type
#print X_test.type
#print y_test.type

model.fit(X_train, y_train, epochs=40, validation_data = (X_test, y_test), verbose=0, batch_size=10)

#model.fit(trainingData, trainingLabelList, verbose = 2, batch_size=1000, epochs=40, shuffle=True, validation_data=(validationData, validationLabelList), callbacks=[checkpointer,earlystopper])


loss, accuracy = model.evaluate(X_train, y_train, verbose = False)
print 'Training accuracy: {:.4f}'.format(accuracy)

loss, accuracy = model.evaluate(X_test, y_test, verbose = False)
print 'Testing accuracy: {:.4f}'.format(accuracy)

