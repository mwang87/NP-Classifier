

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
    Xb = layers.Dense(4096, activation = 'relu')(input_b)
    Xb = layers.BatchNormalization()(Xb)
    Xb = layers.Dense(2048, activation = 'relu')(Xb)
    Xb = layers.BatchNormalization()(Xb)
    Xb = layers.Dense(1024, activation = 'relu')(Xb)
    
    Xf = layers.Dense(2048, activation = 'relu')(input_f)
    Xf = layers.BatchNormalization()(Xf)
    Xf = layers.Dense(1024, activation = 'relu')(Xf)
    Xf = layers.BatchNormalization()(Xf)
    Xf = layers.Dense(512, activation = 'relu')(Xf)
    
    X = layers.Concatenate()([Xb,Xf])
    X = layers.BatchNormalization()(X)
    X = layers.Dense(1536, activation = 'relu')(X)
    X = layers.Dropout(0.2)(X)
    
    output1 = layers.Dense(num, activation = 'sigmoid', name = 'Class')(X)

    model = keras.Model(inputs = [input_f,input_b], outputs = output1)
    
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00001),loss=['binary_crossentropy'],metrics=['cosine_proximity',top_k_categorical_accuracy])
    return model

