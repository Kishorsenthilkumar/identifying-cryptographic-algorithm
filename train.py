import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from sklearn.metrics import classification_report


X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')


model = Sequential()
model.add(Embedding(input_dim=1000, output_dim=128, input_length=128))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(set(y_train)), activation='softmax'))  # Dynamic number of algorithms


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

model.save('model/model.h5')


y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1)


print(classification_report(y_test, y_pred_classes))
