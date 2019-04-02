import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model, load_model

def main():
	dataset = pd.read_csv('../data/book_rating_data/ratings.csv')
	#print dataset.head()
	#print dataset.shape
	
	train, test = train_test_split(dataset, test_size = 0.2, random_state = 42)

	book_data = np.array(list(set(dataset.book_id)))
	print 'First five book ids: '
	print (book_data[:5]) # first five books id

	user = np.array([1 for i in range(len(book_data))]) # creating matrix for first user

	model = load_model('book_recommendation_trained_model_NN.h5')

	predictions = model.predict([user, book_data])
	
	predictions = np.array([a[0] for a in predictions])

	recommended_book_ids = (-predictions).argsort()[:5]

	print 'First five highly rated book ids from user 1: '
	print recommended_book_ids

	print 'Predicted rating of these five books: '
	print predictions[recommended_book_ids]

	books = pd.read_csv('../data/book_rating_data/books.csv')
	print books.head()

	print books[books['id'].isin(recommended_book_ids)]

if __name__ == "__main__":
        main()
