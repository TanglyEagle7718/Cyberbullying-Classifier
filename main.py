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

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(2500,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


"""
CONFIGURE MODEL
Lastly, we need to pick a loss function and an optimizer. Since we are facing 
a binary classification problem and the output of our network is a probability
 (we end our network with a single-unit layer with a sigmoid activation), is
 it best to use the binary_crossentropy loss. It isn't the only viable choice:
     you could use, for instance, mean_squared_error. But crossentropy is 
     usually the best choice when you are dealing with models that output 
     probabilities. Crossentropy is a quantity from the field of Information 
     Theory, that measures the "distance" between probability distributions, 
     or in our case, between the ground-truth distribution and our predictions.
"""
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


"""
validataion data
"""
x_val = x_train[:2500]
partial_x_train = x_train[2500:]

y_val = y_train[:3]
partial_y_train = y_train[3:]    # 15000


"""
TRAIN
"""
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


"""
RESULTS
This object has a member history, which is a dictionary containing data about
 everything that happened during training
It contains 4 entries: one per metric that was being monitored, during 
training and during validation.
"""
results = model.evaluate(partial_x_train, partial_y_train)
print ("train:", results)
results = model.evaluate(x_val, y_val)
print ("validation:", results)
results = model.evaluate(x_test, y_test)
print ("all data", results)

history_dict = history.history
print("history dict.keys():", history_dict.keys())


"""
PLOT LOSS
The dots are the training loss and accuracy, while the solid lines are the 
validation loss and accuracy. Note that your own results may vary slightly 
due to a different random initialization of your network.

As you can see, the training loss decreases with every epoch and the training 
accuracy increases with every epoch. That's what you would expect when running
 gradient descent optimization -- the quantity you are trying to minimize
 should get lower with every iteration. But that isn't the case for the 
 validation loss and accuracy: they seem to peak at the fourth epoch. 
 This is an example of what we were warning against earlier: a model that 
 performs better on the training data isn't necessarily a model that will 
 do better on data it has never seen before. In precise terms, what you are 
 seeing is "overfitting": after the second epoch, we are over-optimizing on 
 the training data, and we ended up learning representations that are specific 
 to the training data and do not generalize to data outside of the training set.

In this case, to prevent overfitting, we could simply stop training after 
three epochs. 
"""
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# =============================================================================
# # "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# 
# plt.show()
# =============================================================================

"""
plot accuracy
"""
fig, ax = plt.subplots()
ax.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
ax.plot(epochs, val_loss, 'b', label='Validation loss')
ax.set(xlabel='Epochs', ylabel='Loss',
       title='Training and validation loss');
ax.legend()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

fig1, ax1 = plt.subplots()
ax1.plot(epochs, acc, 'ro', label='Training acc')
ax1.plot(epochs, val_acc, 'b', label='Validation acc')
ax1.set(xlabel='Epochs', ylabel='Loss',
       title='Training and validation accuracy');
ax1.legend()

plt.show()

"""
predict
"""

testPrediction = model.predict(x_test)


# Fitting K-NN to the Training set
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#classifier.fit(x_train, y_train)

#making predictions on test data
#y_predict = classifier.predict(x_test)

#print("The predict is:",y_predict)

#score=classifier.score(x_test,y_predict)

#print("The Score is:",score)
###


