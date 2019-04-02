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

	#prediction_performance = model.evaluate([test.user_id, test.book_id], test.rating)

	#print str(prediction_performance)

	predictions = model.predict([test.user_id, test.book_id])
	mae_from_prediction = 0.0

	for i in range(len(test.user_id)):
		mae_from_prediction += abs(predictions[i][0] - test.rating.iloc[i])
	
	print 'Mean absolute error is: ' + str(mae_from_prediction / len(test.user_id))

if __name__ == "__main__":
        main()
