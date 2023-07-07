# %%
# !! {"metadata":{
# !!   "id": "ELzYjgrYY8x5"
# !! }}
"""
##### Copyright 2022 The TensorFlow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
"""

# %%
# !! {"metadata":{
# !!   "id": "jQc0tqMFKBmv"
# !! }}
"""
# Image aesthetic scoring with MUSIQ models

In this colab, you can try any of the 4 [MUSIQ](https://ai.googleblog.com/2022/10/musiq-assessing-image-aesthetic-and.html) models to score the aesthetic of an image.

These models have a specific characteristic that it can handle images in any size or aspect ratio, making it not necessary to do any preprocessing on the image

"""

# %%
# !! {"metadata":{
# !!   "id": "2oNh0fGUlHOF"
# !! }}
import tensorflow as tf
import tensorflow_hub as hub

import requests
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

# %%
# !! {"metadata":{
# !!   "id": "KypTsmkzoV1y"
# !! }}
"""
# Load Model
"""

# %%
# !! {"metadata":{
# !!   "id": "1wO1PVJqlQxr"
# !! }}
selected_model = 'koniq' #@param ['spaq', 'koniq', 'paq2piq', 'ava']

NAME_TO_HANDLE = {
    # Model trained on SPAQ dataset: https://github.com/h4nwei/SPAQ
    'spaq': 'https://tfhub.dev/google/musiq/spaq/1',

    # Model trained on KonIQ-10K dataset: http://database.mmsp-kn.de/koniq-10k-database.html
    'koniq': 'https://tfhub.dev/google/musiq/koniq-10k/1',

    # Model trained on PaQ2PiQ dataset: https://github.com/baidut/PaQ-2-PiQ
    'paq2piq': 'https://tfhub.dev/google/musiq/paq2piq/1',

    # Model trained on AVA dataset: https://ieeexplore.ieee.org/document/6247954
    'ava': 'https://tfhub.dev/google/musiq/ava/1',
}

#model_handle = NAME_TO_HANDLE[selected_model]
#model = hub.load(model_handle)
#predict_fn = model.signatures['serving_default']
#
#print(f'loaded model {selected_model} ({model_handle})')

saved_model_dir = selected_model + "/saved_model"
model = tf.saved_model.load(saved_model_dir)
predict_fn = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

# %%
# !! {"metadata":{
# !!   "id": "nuIo92Y-oXSZ"
# !! }}
"""
# Util functions

Functions to download an image and to show it.
"""

# %%
# !! {"metadata":{
# !!   "id": "_BQJLaepC9YV"
# !! }}
def load_image_from_url(img_url):
  """Returns an image with shape [1, height, width, num_channels]."""
  user_agent = {'User-agent': 'Colab Sample (https://tensorflow.org)'}
  response = requests.get(img_url, headers=user_agent)
  image_bytes = BytesIO(response.content)
  image = Image.open(image_bytes)
  return image, response.content

def show_image(image, title=''):
  image_size = image.size
  plt.imshow(image)
  plt.axis('on')
  plt.title(title)
  plt.show()

# %%
# !! {"metadata":{
# !!   "id": "tksog9MdI3li"
# !! }}
"""
# Load an image

The prediction will run directly on the bytes loaded, no preprocessing is needed
"""

# %%
# !! {"metadata":{
# !!   "id": "P0oULq_NnxNg"
# !! }}
image_url = 'https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgr0DKaAoO6qTrJo3hXP8UM3D4AB8gQeNI22Q2QphBVGgn-5v84tjhH3ZWTlGtlUoPdlcx54dM93Qi04MuN7eBbj9WlT8Qxy6B2Us4kcn_53FH28MnTtGCzMPhjCVGIgXRL8ZEMeO-7iue7sNEGxBtgx2bI-eKDQAondM8Dfjb1FaybFgUQji4UU9-0vQ/s1024/image9.png' #@param {type: 'string'}

image, image_bytes = load_image_from_url(image_url)

show_image(image)

# %%
# !! {"metadata":{
# !!   "id": "3-i-cBoEJMn_"
# !! }}
"""
# Run Prediction on a Single Image
"""

# %%
# !! {"metadata":{
# !!   "id": "TuE4kKpvokQZ"
# !! }}
prediction = predict_fn(tf.constant(image_bytes))
print("predicted MOS: ", prediction)

# %%
# !! {"main_metadata":{
# !!   "accelerator": "GPU",
# !!   "colab": {
# !!     "collapsed_sections": [],
# !!     "last_runtime": {
# !!       "build_target": "//learning/deepmind/public/tools/ml_python:ml_notebook",
# !!       "kind": "private"
# !!     },
# !!     "provenance": [
# !!       {
# !!         "file_id": "1i9hws3sIbIAzPFW-XAiRQO9PgVoYDXc3",
# !!         "timestamp": 1667387207919
# !!       }
# !!     ]
# !!   },
# !!   "gpuClass": "standard",
# !!   "kernelspec": {
# !!     "display_name": "Python 3",
# !!     "name": "python3"
# !!   },
# !!   "language_info": {
# !!     "name": "python"
# !!   }
# !! }}
