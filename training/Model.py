

#Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
def top_k_categorical_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)
#Build model

def model_build(num): # num = number of categories
    input_f = layers.Input(shape=(2048,))
    input_b = layers.Input(shape=(4096,))
    input_fp = layers.Concatenate()([input_f,input_b])
    
    X = layers.Dense(6144, activation = 'relu')(input_fp)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(3072, activation = 'relu')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(1536, activation = 'relu')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(1536, activation = 'relu')(X)
    X = layers.Dropout(0.2)(X)
    output = layers.Dense(num, activation = 'sigmoid')(X)
    model = keras.Model(inputs = [input_f,input_b], outputs = output)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00001),loss=['binary_crossentropy'],metrics=['cosine_proximity',top_k_categorical_accuracy])
    return model

