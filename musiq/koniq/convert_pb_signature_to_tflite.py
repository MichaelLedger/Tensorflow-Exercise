#import tensorflow as tf

## Path to the input .pb file
#input_path = "saved_model.pb"
#
## Path to the output .tflite file
#output_path = "saved_model.tflite"
#
## Load the model from the .pb file
#graph_def = tf.compat.v1.GraphDef()
#with tf.io.gfile.GFile(input_path, "rb") as f:
#    graph_def.ParseFromString(f.read())
#
## Convert the model to TFLite format
#converter = tf.compat.v1.lite.TFLiteConverter.from_session(graph_def)
#tflite_model = converter.convert()
#
## Save the TFLite model to disk
#with open(output_path, "wb") as f:
#    f.write(tflite_model)

import tensorflow as tf
import tensorflow_hub as hub

import requests
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
# tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
# tf.contrib.lite -> tf.lite
saved_model_dir = "saved_model"
#converter = tf.lite.TFLiteConverter.from_saved_model(
#    saved_model_dir, signature_keys=['serving_default'])
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.experimental_new_converter = True
#converter._experimental_lower_tensor_list_ops = True
#converter.allow_custom_ops = True
#converter.target_spec.supported_ops = [
#    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
#]
#tflite_model = converter.convert()

def resize_by_axis(image, dim_1, dim_2, ax):
    # Make permutation of dimensions to put ax first
    dims = tf.range(tf.rank(image))
    perm1 = tf.concat([[ax], dims[:ax], dims[ax + 1:]], axis=0)
    # Transpose to put ax dimension first
    image_tr = tf.transpose(image, perm1)
    # Resize
    resized_tr = tf.image.resize_images(image_tr, [dim_1, dim_2],
                                        method=1, align_corners=True)
    # Make permutation of dimensions to put ax in its place
    perm2 = tf.concat([dims[:ax] + 1, [0], dims[ax + 1:]], axis=0)
    # Transpose to put ax in its place
    resized = tf.transpose(resized_tr, perm2)
    return resized

model = tf.saved_model.load(saved_model_dir)
#signatures
# Print the signatures from the converted model
#signatures = interpreter.get_signature_list()
print('Signatures:', model.signatures)
#signature = model.signatures['serving_default']
signature = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
print('signature:', signature)
# Get the first input tensor
#AttributeError: 'list' object has no attribute 'keys'
#input_tensor_name = signature.inputs.keys()[0]
#input_tensor_name = "image_bytes_tensor"
input_tensor = signature.inputs[0]
print('input_tensor:', input_tensor)
#input_tensor = signature.inputs[0]

#print("First input tensor name:", input_tensor_name)
print("First input tensor shape:", input_tensor.shape)
print("First input tensor dtype:", input_tensor.dtype)

#Resize the dynamic shape input shape to an image type.[1, 224, 224, 3]
#image_height = 224
#image_width = 224
#resized_input = tf.image.resize(input_tensor, [1, image_height, image_width])
#resized_input = resize_by_axis(input_tensor, dim_1=image_width, dim_2=image_height, ax=3)
#resized_input = tf.image.resize(input_tensor, [1, 224, 224, 3])

# Resize the input tensor to the desired shape
#input_tensor: Tensor("image_bytes_tensor:0", shape=(), dtype=string)
#ValueError: 'images' must have either 3 or 4 dimensions.
#resized_input = tf.image.resize(input_tensor, [224, 224])

# Add an extra dimension to represent the batch size of 1
#resized_input = tf.expand_dims(resized_input, axis=0)

# Set the number of channels to 3
#resized_input = tf.tile(resized_input, [1, 1, 1, 3])

# Print the shape of the resized tensor
#print(resized_input.shape)

# Replace the dynamic shape resize operation with a specific input shape
#input_shape = (224, 224, 3)  # Example input shape
#signature.build(tf.TensorShape([None] + list(input_shape)))

converter = tf.lite.TFLiteConverter.from_concrete_functions([signature])
#WARNING:absl:Optimization option OPTIMIZE_FOR_SIZE is deprecated, please use optimizations=[Optimize.DEFAULT] instead.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter._experimental_lower_tensor_list_ops = True
converter.allow_custom_ops = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
#AttributeError: 'TFLiteConverterV2' object has no attribute 'resize_tensor_input'
#converter.resize_tensor_input(converter.get_input_details()[0]['index'], [1, 224, 224, 3])
tflite_model = converter.convert()

TFLITE_FILE_PATH="koniq.tflite"

fo = open(
    TFLITE_FILE_PATH, "wb")
fo.write(tflite_model)
fo.close

#dynamic shape
# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
# Resize input shape for dynamic shape model and allocate tensor
interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'], [1, 224, 224, 3])
interpreter.allocate_tensors()
# Get input and output tensors.
# Print input shape and type
inputs = interpreter.get_input_details()
print('{} input(s):'.format(len(inputs)))
for i in range(0, len(inputs)):
    print('{} {}'.format(inputs[i]['shape'], inputs[i]['dtype']))

# Print output shape and type
outputs = interpreter.get_output_details()
print('\n{} output(s):'.format(len(outputs)))
for i in range(0, len(outputs)):
    print('{} {}'.format(outputs[i]['shape'], outputs[i]['dtype']))
    
#exit()
#print("code should never come to here!")

# Create a TFLite converter
#converter = tf.lite.TFLiteConverter.from_concrete_functions([interpreter])
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.experimental_new_converter = True
#converter._experimental_lower_tensor_list_ops = True
#converter.allow_custom_ops = True
#converter.target_spec.supported_ops = [
#    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
#]
#converter.target_spec.supported_types = [tf.float32]
#resize_tflite_model = converter.convert()
#
#with open("koniq_resize.tflite", "wb") as f:
#    f.write(resize_tflite_model)
#    f.close

#inference sample
image_url = 'https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgr0DKaAoO6qTrJo3hXP8UM3D4AB8gQeNI22Q2QphBVGgn-5v84tjhH3ZWTlGtlUoPdlcx54dM93Qi04MuN7eBbj9WlT8Qxy6B2Us4kcn_53FH28MnTtGCzMPhjCVGIgXRL8ZEMeO-7iue7sNEGxBtgx2bI-eKDQAondM8Dfjb1FaybFgUQji4UU9-0vQ/s1024/image9.png' #@param {type: 'string'}

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

image, image_bytes = load_image_from_url(image_url)

show_image(image)

#Detected at node 'PartitionedCall' defined at (most recent call last):
#Node: 'PartitionedCall'
#could not find registered platform with id: 0x12d7b4b60
#     [[{{node PartitionedCall}}]] [Op:__inference_signature_wrapper_25722]
prediction = signature(tf.constant(image_bytes))
print("predicted MOS: ", prediction)

print("exit")
exit()

# Create an EagerTensor object
eager_tensor = tf.constant(image_bytes)
print("eager_tensor:", eager_tensor)

#string_input = tf.keras.Input(shape=(1,), name=header, dtype='string')

def utf8len(s):
    return len(s.encode('utf-8'))

interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()
# Resize input shape for dynamic shape model and allocate tensor
#https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter
#resize_tensor_input(input_index, tensor_size, strict=False)
#interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'], [num_test_images, image.height, image.width, 3])
#interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'], tf.shape(eager_tensor))
#interpreter.resize_tensor_input(interpreter.get_input_details()[0]['index'], (1,))
#interpreter.allocate_tensors()

# Print input shape and type
input_details = interpreter.get_input_details()
print('{} input(s):'.format(len(input_details)))
for i in range(0, len(input_details)):
    print('{} {}'.format(input_details[i]['shape'], input_details[i]['dtype']))

# Print output shape and type
outputs = interpreter.get_output_details()
print('\n{} output(s):'.format(len(outputs)))
for i in range(0, len(outputs)):
    print('{} {}'.format(outputs[i]['shape'], outputs[i]['dtype']))
    
#1 input(s):
#[   1  768 1024    3] <class 'numpy.bytes_'>
#
#1 output(s):
#[] <class 'numpy.float32'>

#interpreter.set_tensor(input_details[0]['index'], tf.constant(image_bytes))
#numpy_array = np.frombuffer(image_bytes, dtype=np.uint8)
#ValueError: Cannot set tensor: Got value of type UINT8 but expected type STRING for input 0, name: image_bytes_tensor
#interpreter.set_tensor(input_details[0]['index'], numpy_array)
#eager_tensor = tf.convert_to_tensor(image_bytes, dtype=tf.uint8)
# Convert the EagerTensor to a NumPy array
#numpy_array = eager_tensor.numpy()
# Convert the UINT8 tensor to a STRING tensor
#string_tensor = tf.strings.as_string(eager_tensor)
#numpy_array = np.array(tf.io.decode_raw(image_bytes, tf.uint8))
#string_numpy_array = tf.cast(numpy_array, tf.string)
#image_string = tf.io.encode_jpeg(image_bytes)
#ValueError: Cannot set tensor: Got value of type UINT8 but expected type STRING for input 0, name: image_bytes_tensor
print("start set_tensor")
#ValueError: Cannot set tensor: Dimension mismatch. Got 1 but expected 0 for input 0.
#interpreter.set_tensor(input_details[0]['index'], [eager_tensor])
interpreter.set_tensor(input_details[0]['index'], eager_tensor)
print("start invoke")
interpreter.invoke()
#prediction = signature(tf.constant(image_bytes))
#print("predicted MOS: ", prediction)
