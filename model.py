# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 16:32:38 2022

@author: sjurm
"""

import numpy as np
import tensorflow as tf

from keras.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from tensorflow.python.ops.nn_ops import dropout
from tensorflow.python.ops.gen_nn_ops import relu

actions = [
    '안녕하세요',
    '먹다',
    '밥',
    '만나다'
    
]

data = np.concatenate([
    
    np.load('dataset2/seq_hi_1_1661663755.npy'),
    np.load('dataset2/seq_hi_2_1661663809.npy'),
    np.load('dataset2/seq_hi_3_1661663843.npy'),
    np.load('dataset2/seq_hi_4_1661663887.npy'),
    np.load('dataset2/seq_eat_1_1661663981.npy'),
    np.load('dataset2/seq_eat_2_1661664034.npy'),
    np.load('dataset2/seq_eat_3_1661664070.npy'),
    np.load('dataset2/seq_eat_4_1661664107.npy'),
    np.load('dataset2/seq_bob_1_1661664188.npy'),
    np.load('dataset2/seq_bob_2_1661664238.npy'),
    np.load('dataset2/seq_bob_3_1661664273.npy'),
    np.load('dataset2/seq_bob_4_1661664322.npy'),
    np.load('dataset2/seq_meet_1_1661664416.npy'),
    np.load('dataset2/seq_meet_2_1661664458.npy'),
    np.load('dataset2/seq_meet_3_1661664486.npy'),
    np.load('dataset2/seq_meet_4_1661664522.npy'),
    
    
], axis=0)   
data.shape

data2 = np.concatenate([
    np.load('dataset2/seq_hi_5_1661663923.npy'),
    np.load('dataset2/seq_eat_5_1661664150.npy'),
    np.load('dataset2/seq_bob_5_1661664366.npy'),
    np.load('dataset2/seq_meet_5_1661664574.npy')
    
    
    
], axis=0)  
data.shape

x_data = data[:,:,:-1]
x_data2 = data2[:,:,:-1]
labels = data[:,0,-1]
labels2 = data2[:, 0, -1]
y_data=tf.keras.utils.to_categorical(labels, num_classes=len(actions))

y_data2 =tf.keras.utils.to_categorical(labels2, num_classes=len(actions))

x_data = x_data.astype(np.float32)
y_data = labels.astype(np.float32)

x_data2 = x_data2.astype(np.float32)
y_data2 = y_data2.astype(np.float32)

x_train = x_data


x_val = x_data2
y_train = y_data

y_val = y_data2


# model2 = tf.keras.models.Sequential([
#    tf.keras.layers.Input(shape=(30,432),name='input'),
#    tf.keras.layers.LSTM(20, time_major=False, return_sequences=True),
#    tf.keras.layers.Flatten(),
#    tf.keras.layers.Dense(3, activation=tf.nn.softmax, name='output')
# ])

model2 = tf.keras.models.Sequential([
   tf.keras.layers.Input(shape=(30,368),name='input'),
   tf.keras.layers.LSTM(64, time_major=False, return_sequences=True),
   tf.keras.layers.Dropout(0.3),
   tf.keras.layers.Dense(32, activation=tf.nn.relu),
   tf.keras.layers.Dense(32, activation=tf.nn.relu),
   tf.keras.layers.Dropout(0.3),
   tf.keras.layers.Dense(64, activation=tf.nn.relu),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(4, activation=tf.nn.softmax, name='output')
])
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.summary()

_EPOCHS = 200


model2.fit(x_train, labels, epochs=_EPOCHS)

run_model = tf.function(lambda x: model2(x))
# This is important, let's fix the input size.
BATCH_SIZE = 1
STEPS = 30
INPUT_SIZE = 368
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model2.inputs[0].dtype))

# model directory.
MODEL_DIR = "AAA"
model2.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open("AAAA.tflite", "wb").write(tflite_model)
