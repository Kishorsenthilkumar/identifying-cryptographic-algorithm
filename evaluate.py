import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np

model = tf.keras.models.load_model('model/model.h5')


X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

y_pred = model.predict(X_test)
y_pred_classes = tf.argmax(y_pred, axis=1)


print(classification_report(y_test, y_pred_classes))
