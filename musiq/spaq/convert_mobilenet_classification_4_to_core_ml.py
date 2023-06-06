# Tested with TensorFlow 2.6.2
import tensorflow as tf
import tensorflow_hub as tf_hub
import numpy as np

#1. Download the MobileNet SavedModel directory from imagenet in TensorFlow Hub.
#model = tf.keras.Sequential([
#        tf.keras.layers.InputLayer(input_shape=(1, 3, 224, 224)),
#        tf_hub.KerasLayer(
#          "https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/classification/4"
#        )
#])
model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(192, 192, 3)),
        tf_hub.KerasLayer(
          "https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/classification/4"
        )
])

# Define the shape of the input tensor
#input_shape = [None, None, 3]

# Define the placeholder for the input tensor
#image_bytes = tf.placeholder(tf.uint8, shape=input_shape)

model.build([1, 192, 192, 3])  # Batch input shape.
#model.build(input_shape)

#2. Load the model as a Keras model, and ensure that it is loaded correctly by applying a prediction call.
# random input data to check that predict works
# Define the local path to the image file
random_image_path = "/Users/gavinxiang/Downloads/Tensorflow-Exercise/musiq/tmp/image9.png"

# Read the image file as a byte string
#x = tf.io.read_file(random_image_path)

x = np.random.rand(1, 192, 192, 3)

tf_out = model.predict([x])

import coremltools as ct

#3. Convert the model to an ML program without specifying the input type, in order to generate a multidimensional array (MLMultiArray) input for convenience in checking predictions:
# convert to Core ML and check predictions
mlmodel = ct.convert(model, convert_to="mlprogram")

#4. Since the model operates on images, convert with the image input type before saving the model as a Core ML model package:
coreml_out_dict = mlmodel.predict({"input_1":x})
coreml_out = list(coreml_out_dict.values())[0]
np.testing.assert_allclose(tf_out, coreml_out, rtol=1e-2, atol=1e-1)

# convert to an image input Core ML model
# mobilenet model expects images to be normalized in the interval [-1,1]
# hence bias of -1 and scale of 1/127
mlmodel = ct.convert(model, convert_to="mlprogram",
                    inputs=[ct.ImageType(bias=[-1,-1,-1], scale=1/127)])

#mlmodel.save("koniq.mlpackage")
mlmodel.save("mobilenet_classification_4.mlpackage")
