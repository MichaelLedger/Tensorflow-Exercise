#import tfcoreml
#
#tfcoreml.convert(tf_model_path='saved_model.pb',
#                 mlmodel_path='koniq.mlmodel',
##                 output_feature_names=['softmax:0'],  # name of the output tensor (appended by ":0")
##                 input_name_shape_dict={'input:0': [1, 227, 227, 3]},  # map from input tensor name (placeholder op in the graph) to shape
#                 minimum_ios_deployment_target='13') # one of ['12', '11.2']

#https://coremltools.readme.io/docs

#https://coremltools.readme.io/docs/unified-conversion-api

#https://stackoverflow.com/questions/59657166/convert-frozen-model-pb-to-savedmodel

#https://medium.com/@jianshi_94445/convert-a-tensorflow-model-to-coreml-model-using-tfcoreml-8ce157f1bc3b

#https://stackoverflow.com/questions/57614436/od-graph-def-tf-graphdef-attributeerror-module-tensorflow-has-no-attribut

#https://stackoverflow.com/questions/61883290/to-load-pb-file-decodeerror-error-parsing-message

#import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

#with tf.io.gfile.GFile('saved_model.pb', 'rb') as f:
#    graph_def = tf.compat.v1.GraphDef()
#    graph_def.ParseFromString(f.read())

#import tensorflow as tf
#import sys
#from tensorflow.python.platform import gfile
#from tensorflow.core.protobuf import saved_model_pb2
#from tensorflow.python.util import compat
#
#with tf.compat.v1.Session() as sess:
#    model_filename ='saved_model.pb'
#    with gfile.FastGFile(model_filename, 'rb') as f:
#        data = compat.as_bytes(f.read())
#        sm = saved_model_pb2.SavedModel()
#        sm.ParseFromString(data)
#        g_in = tf.import_graph_def(sm.meta_graphs[0].graph_def)

import tensorflow as tf
import coremltools as ct

#model = ct.convert(
#                   graph_def,
#                   inputs=[ct.TensorType(name='input', shape=input_shape)],
#                   outputs=[ct.TensorType(name='output', shape=output_shape)]
#                   )

#ValueError: Unable to determine the type of the model, i.e. the source framework. Please provide the value of argument "source", from one of ["tensorflow", "pytorch", "milinternal"]. Note that model conversion requires the source package that generates the model. Please make sure you have the appropriate version of source package installed. E.g., if you're converting model originally trained with TensorFlow 1.14, make sure you have `tensorflow==1.14` installed.

saved = tf.keras.models.load_model('saved_model/saved_model.pb')

model = ct.convert(
                   model=New_Model,
                   source='tensorflow'
                   )

model.save('koniq.mlmodel')
