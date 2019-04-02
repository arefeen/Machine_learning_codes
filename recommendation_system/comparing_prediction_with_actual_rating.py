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

	model = load_model('book_recommendation_trained_model.h5')

	val = 20

	predictions = model.predict([test.user_id.head(val), test.book_id.head(val)])

	mae_cal = 0.0
	for i in range(0, val):
		mae_cal += abs(predictions[i][0] - test.rating.iloc[i])
		print 'Model prediction: ' + str(predictions[i][0]) + ' actual rating: ' + str(test.rating.iloc[i])
	
	print 'Mean absolute error: ' + str(mae_cal / val)

if __name__ == "__main__":
        main()
