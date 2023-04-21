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
import numpy as np

# tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
# tf.contrib.lite -> tf.lite
# Convert the model
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()
#latest_checkpoint = tf.compat.v1.train.latest_checkpoint(checkpoint_dir)
saved_model_dir = "saved_model"
model = tf.saved_model.load(saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(
    saved_model_dir, signature_keys=['serving_default'])
# Load the checkpoints from the .npz file

# use the load_weights method of the Keras model
# Path to the checkpoint directory
#checkpoint_dir = "../tmp/musiq-spaq_ckpt.npz"
#converter.load_weights(checkpoint_dir)

# Set the input and output tensors
inputs = model.signatures["serving_default"].inputs
input_tensor = inputs[0]
print('input: {}'.format(input_tensor))
#with tf.Graph().as_default():
#    tf.cast(input_tensor, np.byte)
#    tf.cast(input_tensor, tf.float32)
#    print('input: {}'.format(input_tensor))
#    reshaped_tensor = tf.TensorArray(tf.reshape(input_tensor, [1, 224, 224]))  # Reshape input tensor to [None, 224, 224, 3]
#    reshaped_tensor = tf.TensorArray(tf.reshape(input_tensor, [602112]))
#    tf.expand_dims(tf.math.reduce_mean(input_tensor, axis=-1, keepdims=True), axis=0).shape
#print('input: {}'.format(input_tensor))

input_tensor = tf.constant([602112], dtype=np.byte)
print('input: {}'.format(input_tensor))

outputs = model.signatures["serving_default"].outputs
output_tensor = outputs[0]
print('output: {}'.format(output_tensor))

#converter.inputs = [input_tensor]
#converter.input_array = [input_tensor]
#converter.outputs = [output_tensor]

# Define a representative dataset
#def representative_dataset():
#    # Create a tensor with the same shape and type as the new input tensor
#    input_tensor = tf.ones([1, 224, 224, 3], dtype=tf.float32)
#    yield [input_tensor]

# Replace the input tensor
#converter.representative_dataset = representative_dataset

print('converter.representative_dataset: {}'.format(converter.representative_dataset))

# Override the model.
#this.converter = tf.converter({inputs: [input_tensor], outputs: [output_tensor]});

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True
tflite_model = converter.convert()

# Save the model
#fo = open(
#    "spaq.tflite", "wb")
#fo.write(tflite_model)
#fo.close

with open('spaq.tflite', 'wb') as f:
    f.write(tflite_model)

print('==generate success!==')

#interpreter = tf.lite.Interpreter(model_path="spaq.tflite")
#interpreter.allocate_tensors()
#
## Print input shape and type
#inputs = interpreter.get_input_details()
#print('{} input(s):'.format(len(inputs)))
#for i in range(0, len(inputs)):
#    print('{} {}'.format(inputs[i]['shape'], inputs[i]['dtype']))
#
## Print output shape and type
#outputs = interpreter.get_output_details()
#print('\n{} output(s):'.format(len(outputs)))
#for i in range(0, len(outputs)):
#    print('{} {}'.format(outputs[i]['shape'], outputs[i]['dtype']))

# Call a python script from another python script?
import inspect_tflite
inspect_tflite.inspect()
