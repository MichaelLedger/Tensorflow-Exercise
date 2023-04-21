# Check whether you have installed tensorflow_hub
# conda list
# pip install tensorflow_hub

import tensorflow as tf
import tensorflow_hub as hub

#import requests
from PIL import Image
from io import BytesIO
#
#import matplotlib.pyplot as plt
#import numpy as np

image_path="/Users/gavinxiang/Downloads/Tensorflow-Exercise/musiq/tmp/image.jpeg"

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image
  
#def load_image_from_url(img_url):
#  """Returns an image with shape [1, height, width, num_channels]."""
#  user_agent = {'User-agent': 'Colab Sample (https://tensorflow.org)'}
#  response = requests.get(img_url, headers=user_agent)
#  image_bytes = BytesIO(response.content)
#  image = Image.open(image_bytes)
#  return image, response.content

def load_image(image_path):
    """Loads and preprocesses images."""
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
#    img = tf.io.decode_image(
#        tf.io.read_file(image_path),
#        channels=3, dtype=tf.float32)[tf.newaxis, ...]
#    img = crop_center(img)
#    image_size=(256, 256)
#    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
#    image_bytes = BytesIO(response.content)
    image_bytes = tf.io.read_file(image_path)
    image = Image.open(image_bytes)
    return img

image_bytes = load_image(image_path)
#model = hub.load('https://tfhub.dev/google/musiq/spaq/1')
model = hub.load('../musiq/spaq/saved_model')
predict_fn = model.signatures['serving_default']

aesthetic_score = predict_fn(tf.constant(image_bytes))

#tensorflow.python.framework.errors_impl.InvalidArgumentError: cannot compute __inference_signature_wrapper_25776 as input #0(zero-based) was expected to be a string tensor but is a float tensor [Op:__inference_signature_wrapper_25776]
#aesthetic_score = predict_fn(tf.constant(image_path))

print('mos: {}'.format(aesthetic_score))
