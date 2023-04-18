#https://www.tensorflow.org/api_docs/python/tf/compat/v1/GraphDef

import tensorflow as tf

# Load the graph from the .pb file
#with tf.gfile.GFile('saved_model.pb', 'rb') as f:
#    graph_def = tf.GraphDef()
#    graph_def.ParseFromString(f.read())

#tf.saved_model.load('saved_model')

@tf.function
def f(x):
  return x

graph_def = f.get_concrete_function(1.).graph.as_graph_def()
print(graph_def)

# Convert the graph to a SavedModel
builder = tf.compat.v1.saved_model.builder.SavedModelBuilder('./saved')
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    # Import the graph def into the new graph
    tf.import_graph_def(graph_def, name='')
    # Save the model
    builder.add_meta_graph_and_variables(sess, [tf.compat.v1.saved_model.tag_constants.SERVING])
builder.save()
