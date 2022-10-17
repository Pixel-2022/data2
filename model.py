# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 14:45:11 2022

@author: sjurm
"""

from gc import callbacks
import numpy as np
import os
import tensorflow as tf


#from keras.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from tensorflow.python.ops.nn_ops import dropout
from tensorflow.python.ops.gen_nn_ops import relu
from keras import optimizers
from keras.callbacks import EarlyStopping
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
data=[]
data2=[]
dir_path = "train"
dir_path2 = "test"
actions = []
#data = np.concatenate([np.load('dataset3/seq_가족_F_1663653962.npy')])
np.set_printoptions(threshold=np.inf,linewidth=np.inf)
for (root, directories, files) in os.walk(dir_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_path2 = file_path.split("\\")
        file_path3 = file_path2[1].split(".")
        print(file_path3[0])
        if data==[]:
            data = np.concatenate([
            np.load(file_path)],axis=0)
        else:
            data = np.concatenate([data,
                np.load(file_path)],axis=0)
            
        print(data.shape)
        
        
     
        

for (root2, directories2, files2) in os.walk(dir_path2):
    for file_test in files2:
        file_path_test = os.path.join(root2, file_test)
        file_path2_test = file_path_test.split("\\")
        file_path3_test = file_path2_test[1].split(".")
        action = file_path3_test[0]
        action_split = action[4:]
        action_split2 = action_split.split("_", maxsplit=2)
        #print(action_split2[0])
        #print("action에 넣을거야")
        
        actions.append(action_split2[0])
        if data2==[]:
            data2 = np.concatenate([
            np.load(file_path_test)],axis=0)
        else:
            data2 = np.concatenate([data2,
            np.load(file_path_test)],axis=0)
        
print(data.shape)

x_data = data[:,:,:-1]
x_data2 = data2[:,:,:-1]

labels = data[:,0,-1]
labels2 = data2[:, 0, -1]
#print(labels)
#print(x_data.shape)

y_data=tf.keras.utils.to_categorical(labels, num_classes=len(actions))

y_data2 =tf.keras.utils.to_categorical(labels2, num_classes=len(actions))

x_data = x_data.astype(np.float32)
y_data = labels.astype(np.float32)

x_data2 = x_data2.astype(np.float32)
y_data2 = labels2.astype(np.float32)

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
   tf.keras.layers.Input(shape=(30,524),name='input'),
   tf.keras.layers.LSTM(512, time_major=False, return_sequences=True),
   tf.keras.layers.Dropout(0.3),
   tf.keras.layers.LSTM(256, return_sequence=True),
   tf.keras.layers.Dropout(0.3),
   tf.keras.layers.LSTM(128),
   tf.keras.layers.Dense(128, activation=tf.nn.relu),
   tf.keras.layers.Dropout(0.3),
   tf.keras.layers.Dense(32, activation=tf.nn.relu),
   tf.keras.layers.Dense(32, activation=tf.nn.relu),
   tf.keras.layers.Flatten(),
   tf.keras.layers.Dense(33, activation=tf.nn.softmax, name='output')
])
model2.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.summary()


_EPOCHS = 500

callbacks=[EarlyStopping(monitor='val_loss',patience=20,verbose=0)]
history=model2.fit(x_train, labels, validation_data=(x_val,y_val),batch_size=128, callbacks=callbacks,epochs=_EPOCHS,shuffle=True,verbose=1)

run_model = tf.function(lambda x: model2(x))
# This is important, let's fix the input size.
BATCH_SIZE = 1
STEPS = 30
INPUT_SIZE = 524
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([BATCH_SIZE, STEPS, INPUT_SIZE], model2.inputs[0].dtype))

# model directory.
MODEL_DIR = "AAA"
model2.save(MODEL_DIR, save_format="tf", signatures=concrete_func)

converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open("AAAA.tflite", "wb").write(tflite_model)



import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots(figsize=(16, 10))
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history .history['accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()
