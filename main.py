import keras
keras.__version__
from keras.datasets import imdb
import numpy as np
import sklearn.model_selection as model_selection
from keras import models
from keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('new_cyberbullying_data.csv')

# creating bag of words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2500)

x = cv.fit_transform(df['tweet_text']).toarray()
y = cv.fit_transform(df['cyberbullying_type']).toarray()

print(x.shape)
print(y.shape)

#splitting da data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#scales features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

#making predictions on test data
y_predict = classifier.predict(x_test)

print("The predict is:",y_predict)

score=classifier.score(x_test,y_predict)

print("The Score is:",score)

