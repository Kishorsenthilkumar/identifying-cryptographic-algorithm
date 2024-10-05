import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


df = pd.read_csv('cryptographic_dataset.csv')

le = LabelEncoder()
df['algorithm_used'] = le.fit_transform(df['algorithm_used'])
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['ciphertext'])
X = tokenizer.texts_to_sequences(df['ciphertext'])
X = pad_sequences(X, maxlen=128)  # Set maxlen according to your data

X_train, X_test, y_train, y_test = train_test_split(X, df['algorithm_used'], test_size=0.2, random_state=42)


np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
import pickle


with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
