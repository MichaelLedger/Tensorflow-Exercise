import tensorflow as tf

imported = tf.saved_model.load('../musiq/spaq/saved_model') # /path/to/saved_model
concrete_func = imported.signatures["serving_default"]
concrete_func.inputs[0].set_shape([1])
#tf.reshape(concrete_func.inputs[0], [1])
