import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model


def main():
	dataset = pd.read_csv('../data/book_rating_data/ratings.csv')
	#print dataset.head()
	#print dataset.shape
	
	train, test = train_test_split(dataset, test_size = 0.2, random_state = 42)

	#print train.shape
	#print test.shape
	n_users = len(dataset.user_id.unique())
	#print n_users

	n_books = len(dataset.book_id.unique())
	#print n_books

	book_input = Input(shape = [1], name = "Book_Input")
	book_embedding = Embedding(n_books+1, 5, name = "Book_Embedding")(book_input)
	book_vec = Flatten(name = "Flatten-Books")(book_embedding)

	user_input = Input(shape = [1], name = "User-Input")
	user_embedding = Embedding(n_users+1, 5, name = "User-Embedding")(user_input)
	user_vec = Flatten(name = "Flatten-Users")(user_embedding)

	prod = Dot(name = "Dot-Product", axes = 1)([book_vec, user_vec])
	#den1 = Dense(256, activation = 'relu', name = 'fc1')(prod)
	#dout1 = Dropout(0.3)(den1)
	#den2 = Dense(128, activation = 'relu', name = 'fc2')(dout1)
	#dout2 = Dropout(0.3)(den2)
	#den3 = Dense(64, activation = 'relu', name = 'fc3')(dout2)
	#den4 = Dense(1, activation = 'relu', name = 'fc4')(den3)

	#model = Mode([user_input, book_input], den4)
	model = Model([user_input, book_input], prod)
	model.compile(optimizer = 'adam', loss = 'mean_squared_error')
	history = model.fit([train.user_id, train.book_id], train.rating, epochs = 10, verbose = 1)
	model.save('book_recommendation_trained_model.h5')

	model.evaluate([test.user_id, test.book_id], test.rating)

if __name__ == "__main__":
        main()
