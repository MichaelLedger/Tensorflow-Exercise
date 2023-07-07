import tensorflow as tf
#import tensorflow.compat.v1 as tf
#from tensorflow import __version__
print(tf.__version__)
print(tf)
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.compat.v1.Session(config=config)
#tf.config.list_physical_devices('GPU')
#print("Num of GPUs available: ", len(tf.test.gpu_device_name()))

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.models import Sequential, load_model
#
#model = tf.keras.Sequential()
#model.add(layers.Dense(32, input_dim=784))
#model.add(layers.Activation('relu'))
#model.add(layers.LSTM(17))
#model.add(layers.Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.01), metrics=['accuracy'])


#cifar = tf.keras.datasets.cifar100
#(x_train, y_train), (x_test, y_test) = cifar.load_data()
#model = tf.keras.applications.ResNet50(
#    include_top=True,
#    weights=None,
#    input_shape=(32, 32, 3),
#    classes=100,)
#
#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
#model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
#model.fit(x_train, y_train, epochs=5, batch_size=64)
