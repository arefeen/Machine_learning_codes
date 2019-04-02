import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model, load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
	dataset = pd.read_csv('../data/book_rating_data/ratings.csv')
	#print dataset.head()
	#print dataset.shape
	
	train, test = train_test_split(dataset, test_size = 0.2, random_state = 42)

	model = load_model('book_recommendation_trained_model_NN.h5')

	#prediction_performance = model.evaluate([test.user_id, test.book_id], test.rating)

	#print str(prediction_performance)

	#predictions = model.predict([test.user_id, test.book_id])
	
	book_em = model.get_layer('Book_Embedding')
	book_em_weights = book_em.get_weights()[0]

	book_em_weights = book_em_weights / np.linalg.norm(book_em_weights, axis = 1).reshape((-1, 1))

	print book_em_weights[:7]

	pca = PCA(n_components = 2)
	pca_result = pca.fit_transform(book_em_weights)
	plt.scatter(x=pca_result[:,0], y=pca_result[:,1])
	plt.savefig('book_embedding.png')

if __name__ == "__main__":
        main()
