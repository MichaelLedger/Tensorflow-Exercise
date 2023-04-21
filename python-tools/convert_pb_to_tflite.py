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
# tf.contrib will not be distributed with TensorFlow 2.0, please consider an alternative in non-contrib TensorFlow, a community-maintained repository such as tensorflow/addons, or fork the required code.
# tf.contrib.lite -> tf.lite
saved_model_dir = "../musiq/spaq/saved_model"
converter = tf.lite.TFLiteConverter.from_saved_model(
    saved_model_dir, signature_keys=['serving_default'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter._experimental_lower_tensor_list_ops = True
converter.allow_custom_ops = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

fo = open(
    "../musiq/spaq/spaq.tflite", "wb")
fo.write(tflite_model)
fo.close
