import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = tf.keras.models.load_model('model/model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


MAXLEN = 128

def predict_algorithm(ciphertext):
   
    seq = tokenizer.texts_to_sequences([ciphertext])
    padded_seq = pad_sequences(seq, maxlen=MAXLEN, padding='post')
    
 
    y_pred = model.predict(padded_seq)
    y_pred_class = np.argmax(y_pred, axis=1)
    
   
    algorithm_labels = ["AES", "DES", "3DES", "Blowfish", "RSA", "Fernet"]  
    predicted_algorithm = algorithm_labels[y_pred_class[0]]
    
    return predicted_algorithm

if __name__ == "__main__":
    
    input_ciphertext = input("enter the cipher_text:")
    algorithm = predict_algorithm(input_ciphertext)
    print(f"The identified  cryptographic algorithm is: {algorithm}")
