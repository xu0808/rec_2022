# -*- coding:utf-8 -*-
# Keras版本模型保存与加载

import tensorflow as tf
import numpy as np
print(tf.__version__)


x_train = np.random.random((1000, 32))
y_train = np.random.randint(10, size=(1000, ))
x_val = np.random.random((200, 32))
y_val = np.random.randint(10, size=(200, ))
x_test = np.random.random((200, 32))
y_test = np.random.randint(10, size=(200, ))

def get_uncompiled_model():
    inputs = tf.keras.Input(shape=(32,), name='digits')
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = tf.keras.layers.Dense(10, name='predictions')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])
    return model


model = get_compiled_model()
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))

model.summary()


# 方法一
model.save_weights("adasd.h5")
model.load_weights("adasd.h5")
model.predict(x_test)

# 方法二
# Export the model to a SavedModel
model.save('keras_model_tf_version', save_format='tf')
# Recreate the exact same model
new_model = tf.keras.models.load_model('keras_model_tf_version')
new_model.predict(x_test)

# 方法三
model.save('keras_model_hdf5_version.h5')
new_model = tf.keras.models.load_model('keras_model_hdf5_version.h5')
new_model.predict(x_test)

# 方法四
tf.saved_model.save(model, 'tf_saved_model_version')
restored_saved_model = tf.saved_model.load('tf_saved_model_version')
f = restored_saved_model.signatures["serving_default"]
# f(digits=tf.constant(x_test.tolist()))
# !saved_model_cli show --dir tf_saved_model_version --all