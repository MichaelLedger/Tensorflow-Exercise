import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

New_Model = tf.keras.models.load_model('../koniq') # Loading the Tensorflow Saved Model (PB)
#print(New_Model.summary())

# Saving the Model in H5 Format and Loading it (to check if it is same as PB Format)
tf.keras.models.save_model(New_Model, 'koniq.h5', overwrite=True, save_format=None) # Saving the Model in H5 Format

loaded_model_from_h5 = tf.keras.models.load_model('koniq.h5') # Loading the H5 Saved Model
print(loaded_model_from_h5.summary())

loaded_model_from_h5.save('koniq.h5') # Saving a Keras model by creates a h5 file
