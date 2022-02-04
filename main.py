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

fileName = 'new_cyberbullying_data.csv'
print("fileName: ", fileName)
raw_data = open(fileName, 'rt')
#loadtxt defaults to floats
data = np.loadtxt(raw_data, usecols = (0,1), skiprows = 1, delimiter=",", dtype="str")
x = data[:,0] #from 0 to 2
y = data[:,1] #only column 3

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.75,test_size=0.25, random_state=101)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform (x_test)

(train_data, train_labels), (test_data, test_labels) = (x_train, y_train), (x_test, y_test)
#[max(sequence) for sequence in train_data]
print("train_data[0]:", train_data[0])
print("shape: ", train_labels.shape)
print ("max: ", max([max(sequence) for sequence in train_data]))

# word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

print ("decoded review:", decoded_review)

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

print("x_train[0]:", x_train[0])

# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


"""
validataion data
"""
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]    # 15000

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


results = model.evaluate(partial_x_train, partial_y_train)
print ("train:", results)
results = model.evaluate(x_val, y_val)
print ("validation:", results)
results = model.evaluate(x_test, y_test)
print ("all data", results)

history_dict = history.history
print("history dict.keys():", history_dict.keys())

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