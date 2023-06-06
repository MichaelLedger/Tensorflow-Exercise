# Tested with TensorFlow 2.6.2
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow_hub as tf_hub
import numpy as np
import coremltools as ct

#https://github.com/rcmalli/keras-vggface/issues/73
#from keras.utils.layer_utils import get_source_inputs
#import tensorflow.python.keras.engine

#https://huaweicloud.csdn.net/6380889adacf622b8df89abb.html
#from tensorflow.python.keras.layers import Layer, InputSpec

# or kerassurgeon for standalone Keras
#from kerassurgeon import delete_layer, insert_layer
#https://github.com/BenWhetton/keras-surgeon
#from kerassurgeon.operations import delete_layer, insert_layer, delete_channels

#Note: Failed with:
#in user code:
#
#    File "/Users/gavinxiang/miniconda3/lib/python3.10/site-packages/tensorflow_hub/keras_layer.py", line 234, in call  *
#        result = f()
#
#    ValueError: Tensor Tensor("Placeholder:0", shape=(1,), dtype=float32) is not compatible with the shape this function was traced with. Expected shape (), but got shape (1,).
#
#    If you called get_concrete_function, you may need to pass a tf.TensorSpec(..., shape=...) with a less specific shape, having None on axes which can vary.
#
#
#Call arguments received by layer "keras_layer" (type KerasLayer):
#  • inputs=tf.Tensor(shape=(1,), dtype=float32)
#  • training=None

# Define the shape of the input tensor
#input_shape is set to [None, None, 3] to allow for images of any size.
#input_shape = [None, None, 3]
#input_shape = []
#input_shape=tf.TensorSpec(..., shape=...)
# Define a tensor with unknown shape
#input_shape = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.float32)

#RuntimeError: tf.placeholder() is not compatible with eager execution.
#tf.compat.v1.disable_eager_execution()
#https://stackoverflow.com/questions/58986126/replacing-placeholder-for-tensorflow-v2
#input_shape = tf.compat.v1.placeholder(tf.float32, shape=None)
# Use keras.Input instead of placeholders
#input_shape = tf.keras.Input(name="input_image_bytes", shape=(), dtype=tf.dtypes.float32)
#input_shape = tf.keras.Input(shape=None, dtype=tf.dtypes.string)
#input_shape = tf.keras.layers.Input( dtype=tf.string, shape=(1,))
#input_shape = tf.keras.layers.Input( dtype=tf.string, shape=())
#ValueError: tf.enable_eager_execution must be called at program startup.
#tf.compat.v1.enable_eager_execution()

# https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer
#    raise ValueError("Signature name has to be specified for non-callable "
#ValueError: Signature name has to be specified for non-callable saved models (if not legacy TF1 Hub format).
#ValueError: When using a signature, either output_key or signature_outputs_as_dict=True should be set.

#Call arguments received by layer "keras_layer" (type KerasLayer):
#  • inputs=tf.Tensor(shape=(None,), dtype=float32)
#  • training=None
  
#1. Download the MobileNet SavedModel directory from imagenet in TensorFlow Hub.
old_model = tf.keras.Sequential([#tf.keras.layers.InputLayer(input_shape=()),
                                 #tf.keras.layers.Reshape(target_shape=(1,), input_shape=()),
                                 #tf.keras.layers.Input(shape=(1)),
                                 #tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
                                 #tf.keras.layers.Dense(1),
                                 tf_hub.KerasLayer(
                                                   "https://tfhub.dev/google/musiq/spaq/1",
                                                   signature="serving_default",
                                                   signature_outputs_as_dict=True
                                                   )])
#model = tf.keras.Sequential([
#        tf.keras.layers.InputLayer(input_shape=(192, 192, 3)),
#        tf_hub.KerasLayer(
#          "https://tfhub.dev/google/imagenet/mobilenet_v2_050_192/classification/4"
#        )
#])

#Model: "sequential"
#_________________________________________________________________
# Layer (type)                Output Shape              Param #
#=================================================================
# keras_layer (KerasLayer)    {'output_0': ()}          0
#
#=================================================================
#Total params: 0
#Trainable params: 0
#Non-trainable params: 0
#_________________________________________________________________
#============== output_tensor: {'output_0': <KerasTensor: shape=() dtype=float32 (created by layer 'keras_layer')>}
#============== input_tensor: KerasTensor(type_spec=TensorSpec(shape=<unknown>, dtype=tf.float32, name='keras_layer_input'), name='keras_layer_input', description="created by layer 'keras_layer_input'")

#model = tf.reshape(old_model, [1])

#https://stackoverflow.com/questions/49546922/keras-replacing-input-layer
#https://stackoverflow.com/questions/42187425/how-to-change-input-shape-in-sequential-model-in-keras

# old_model.get_config: {'name': 'sequential', 'layers': [{'class_name': 'InputLayer', 'config': {'batch_input_shape': None, 'dtype': 'float32', 'sparse': False, 'ragged': False, 'name': 'keras_layer_input'}}, {'class_name': 'KerasLayer', 'config': {'name': 'keras_layer', 'trainable': False, 'dtype': 'float32', 'handle': 'https://tfhub.dev/google/musiq/spaq/1', 'signature': 'serving_default', 'signature_outputs_as_dict': True}}]}
def change_model(model, new_input_shape=(1,)):
    # replace input shape of first layer
#    print('============== batch_input_shape_before:', model.layers[0]['config'].batch_input_shape)
    #AttributeError: 'KerasLayer' object has no attribute 'batch_input_shape'. Did you mean: '_build_input_shape'?
#    model.layers[0]['config'].batch_input_shape = new_input_shape
#    model.layers[0]._batch_input_shape = new_input_shape
    model_config = model.get_config()
    model_config["layers"][0]["config"]["batch_input_shape"] = new_input_shape
#    print('============== batch_input_shape_after:', model.layers[0]['config'].batch_input_shape)
    print('============== model_config_modified:', model_config)

    # feel free to modify additional parameters of other layers, for example...
#    model._layers[2].pool_size = (8, 8)
#    model._layers[2].strides = (8, 8)

    # rebuild model architecture by exporting and importing via json
#    new_model = tf.keras.models.model_from_json(model.to_json(), custom_objects={'KerasLayer':tf_hub.KerasLayer})

    #https://www.tensorflow.org/api_docs/python/tf/keras/models/model_from_config
    ## for a Functional API model
    #tf.keras.Model().from_config(model.get_config())
    #
    ## for a Sequential model
    #tf.keras.Sequential().from_config(model.get_config())
    new_model = tf.keras.Sequential().from_config(model_config, custom_objects={'KerasLayer':tf_hub.KerasLayer})
    new_model.summary()

    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    # test new model on a random input image
#    X = np.random.rand(10, 40, 40, 3)
#    y_pred = new_model.predict(X)
#    print(y_pred)

    return new_model
    
old_model.build(input_shape=())

print('============== old_model.get_config:', old_model.get_config())
old_model.summary()
old_model.get_weights()
print('============== old_input_tensor:', old_model.input)
print('============== old_output_tensor:', old_model.output)

model = change_model(old_model)

#"tfkerassurgeon" is no longer being maintained and does not work for the current TensorFlow version (2.5.0). On the other hand, "kerassurgeon" DOES support Keras through TensorFlow (i.e tensorflow.keras), not just standalone Keras.
#old_model = delete_layer(model.layers[0])
# inserts before layer 0
#my_input_tensor = Input(input_shape=(1, 3, 224, 224))
#model = insert_layer(old_model.layers[0], my_input_tensor)

#model.build(input_shape=(1,))

model.summary()
model.get_weights()

#model.layers.pop(0)
#model.summary()
#model.get_weights()
#
##newInput = Input(batch_shape=(0,299,299,3))    # let us say this new InputLayer
#newInput = tf.keras.layers.InputLayer(input_shape=(1,))
#newOutputs = model(newInput)
#newModel = Model(newInput, newOutputs)
#
#newModel.summary()
#model.summary()

#https://stackoverflow.com/questions/49546922/keras-replacing-input-layer
#https://www.tensorflow.org/api_docs/python/tf/keras/models/clone_model
# given model: tf.keras.Model
#old_input_layer = model.get_layer(index=0)
#new_input_layer = tf.keras.layers.Input(
#        batch_size=1,  # new desired batch size
#        shape=old_input_layer.input.shape[1:],  # [1:], because we specify the batch size above
#        dtype=old_input_layer.dtype,
#        name=old_input_layer.name,
#    )
#new_model = tf.keras.models.clone_model(old_model, new_input_layer)

print('============== model.get_config:', model.get_config())

#https://www.tensorflow.org/guide/keras/serialization_and_saving#configuration_of_a_sequential_model_or_functional_api_model
model_config = model.get_config()
#model_config["layers"][0]["config"]["batch_input_shape"] = (None, 1)
#Call arguments received by layer "keras_layer" (type KerasLayer):
#  • inputs=tf.Tensor(shape=(None, 1), dtype=float32)
#  • training=None
#new_model = tf.keras.Model.from_config(model_config)
new_model = tf.keras.Sequential.from_config(model_config, custom_objects={'KerasLayer':tf_hub.KerasLayer})

#ValueError: Unknown layer: 'KerasLayer'. Please ensure you are using a `keras.utils.custom_object_scope` and that this object is included in the scope. See https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object for details.

new_model.summary()
new_model.get_weights()

# Define the placeholder for the input tensor
#image_bytes = tf.placeholder(tf.uint8, shape=input_shape)

#ValueError: Tensor Tensor("Placeholder:0", shape=(1,), dtype=float32) is not compatible with the shape this function was traced with. Expected shape (), but got shape (1,).

#If you called get_concrete_function, you may need to pass a tf.TensorSpec(..., shape=...) with a less specific shape, having None on axes which can vary.
    
#Call arguments received by layer 'keras_layer' (type KerasLayer):
#  • inputs=tf.Tensor(shape=(1,), dtype=float32)
#  • training=None

#https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
#model.add(tf.keras.Input(shape=(16,)))
#model.add(input_shape)
#model.add(tf.keras.layers.InputLayer(input_shape=(1,), dtype=tf.string))

#    raise ValueError(
#ValueError: Model <keras.engine.sequential.Sequential object at 0x29f7f0610> cannot be saved because the input shapes have not been set. Usually, input shapes are automatically determined from calling `.fit()` or `.predict()`. To manually set the shapes, call `model.build(input_shape)`.
#model.build([1, 192, 192, 3])  # Batch input shape.
#model.build(input_shape)
#model.build((1,))
#model.build((1,))

# reshape the input tensor
#input_tensor = model.input
#
#new_input_shape = (1,)
#new_input_tensor = tf.keras.reshape(input_tensor, new_input_shape)
#
## create a new model with the reshaped input tensor
#new_model = keras.models.Model(new_input_tensor, model.output)

print('============== input_tensor:', model.input)
print('============== output_tensor:', model.output)

print('============== new_input_tensor:', new_model.input)
print('============== new_output_tensor:', new_model.output)

#2. Load the model as a Keras model, and ensure that it is loaded correctly by applying a prediction call.
# random input data to check that predict works
# Define the local path to the image file
random_image_path = "/Users/gavinxiang/Downloads/Tensorflow-Exercise/musiq/tmp/image9.png"

# Read the image file as a byte string
image_bytes = tf.io.read_file(random_image_path)

#x = np.random.rand(1, 192, 192, 3)
#tf.Tensor(b'\x89PNG\r\n\x1...\x00IEND\xaeB`\x82', shape=(), dtype=string)
x=tf.constant(image_bytes)
#print('============== Precited image bytes:', image_bytes)
print('============== Precited contant:', x.dtype)

#tf_out = model.predict([x])

#3. Convert the model to an ML program without specifying the input type, in order to generate a multidimensional array (MLMultiArray) input for convenience in checking predictions:
# convert to Core ML and check predictions

print('============== start to save ml program')
mlmodel = ct.convert(
                     model,
                     convert_to="mlprogram",
                     source="tensorflow",
                     inputs=[
                           ct.TensorType(name="keras_layer_input", shape=x.shape),  # Replace with the correct shape
                           ],
                     compute_precision=ct.precision.FLOAT16
)

# Set the metadata properties
mlmodel.short_description = "Unofficial pytorch implementation of the paper MUSIQ: Multi-Scale Image Quality Transformer (paper link: https://arxiv.org/abs/2108.05997).A pytorch trained model based on ResNet50 weights pretrained on the ImageNet database and the IQA KonIQ-10k dataset."
mlmodel.author = "PlanetArt: GavinXiang"
mlmodel.license = "MIT License."
mlmodel.version = "1.0.0"  # You can set the version number as a string

#4. Since the model operates on images, convert with the image input type before saving the model as a Core ML model package:
#coreml_out_dict = mlmodel.predict({"input_1":x})
#coreml_out = list(coreml_out_dict.values())[0]
#np.testing.assert_allclose(tf_out, coreml_out, rtol=1e-2, atol=1e-1)

# convert to an image input Core ML model
# mobilenet model expects images to be normalized in the interval [-1,1]
# hence bias of -1 and scale of 1/127
#mlmodel = ct.convert(new_model, convert_to="mlprogram",
#                    inputs=[ct.ImageType(bias=[-1,-1,-1], scale=1/127)])

#mlmodel.save("koniq.mlpackage")
mlmodel.save("mobilenet_classification_4.mlpackage")

print('============== save ml program susscess')
