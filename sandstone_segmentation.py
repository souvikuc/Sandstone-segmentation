# -*- coding: utf-8 -*-
"""sandstone segmentation.ipynb


Original file is located at
    https://colab.research.google.com/drive/1BTxGUuPcmXH9dCEmh4YXh7NvOfFbikQM
"""


# Importing all necessary libraries

!pip install segmentation_models

import os
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import sklearn as sk
from tensorflow import keras
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from skimage import color,feature, filters,exposure, transform, measure, io
import segmentation_models as sm


from google.colab import drive
drive.mount('/content/drive')



# Setting the root path

ROOT = '/content/drive/MyDrive/Deep_learning/Data/Segmentation/sandstone'



# Reading the image and mask files

X = io.imread(ROOT+'/images_as_128x128_patches.tif')
y = io.imread(ROOT+'/masks_as_128x128_patches.tif')



# Checking the image and corresponding masks for an image

idx = np.random.randint(0,len(X))
img = X[idx]
truth = y[idx]


plt.figure(figsize=(16,8))
plt.subplot(121)
plt.title('Actual image')
plt.imshow(img[:,:],cmap = 'gray')
plt.axis('off')
plt.subplot(122)
plt.title('Actual mask')
plt.imshow(truth[:,:],cmap = 'gray')
plt.axis('off')
plt.tight_layout()
plt.show()



# Encoding the mask files

lblenc = LabelEncoder()
y_reshaped = y.reshape(-1)
y_reshaped_encoded = lblenc.fit_transform(y_reshaped)
y_encoded = y_reshaped_encoded.reshape((1600,128,128))



# Expanding the dimensions to feed into the CNN 

X = np.expand_dims(X, axis = 3)
X = keras.utils.normalize(X, axis= 1)

y = np.expand_dims(y_encoded, axis= 3)



# Splitting the data into train and test sets

X_train, X_test,y_train,y_test=train_test_split(X,y_encoded,test_size=0.2)



# Creating the one-hot encoded labels

y_train_cat = keras.utils.to_categorical(y_train,4)
y_test_cat = keras.utils.to_categorical(y_test,4)



# The dataset is imbalanced. Computing the class weights to make it balanced during training

class_weights = sk.utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_reshaped_encoded), y = y_reshaped_encoded)
class_weights



# Defining the U-Net model for segmentation

inputs = keras.layers.Input(shape=(128,128,1))

cnv1 = keras.layers.Conv2D(32, 3, activation = 'relu', padding='same')(inputs)
cnv1 = keras.layers.Dropout(0.2)(cnv1)
cnv1 = keras.layers.Conv2D(32, 3, activation = 'relu', padding='same')(cnv1)
cnv1 = keras.layers.BatchNormalization()(cnv1)
pool1 = keras.layers.MaxPooling2D()(cnv1)


cnv2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding='same')(pool1)
cnv2 = keras.layers.Dropout(0.2)(cnv2)
cnv2 = keras.layers.Conv2D(64, 3, activation = 'relu', padding='same')(cnv2)
cnv2 = keras.layers.BatchNormalization()(cnv2)
pool2 = keras.layers.MaxPooling2D()(cnv2)


cnv3 = keras.layers.Conv2D(128, 3, activation = 'relu', padding='same')(pool2)
cnv3 = keras.layers.Dropout(0.2)(cnv3)
cnv3 = keras.layers.Conv2D(128, 3, activation = 'relu', padding='same')(cnv3)
cnv3 = keras.layers.BatchNormalization()(cnv3)
pool3 = keras.layers.MaxPooling2D()(cnv3)


cnv4 = keras.layers.Conv2D(256, 3, activation = 'relu', padding='same')(pool3)
cnv4 = keras.layers.Dropout(0.2)(cnv4)
cnv4 = keras.layers.Conv2D(256, 3, activation = 'relu', padding='same')(cnv4)
cnv4 = keras.layers.BatchNormalization()(cnv4)
pool4 = keras.layers.MaxPooling2D()(cnv4)


cnv5 = keras.layers.Conv2D(512, 3, activation = 'relu', padding='same')(pool4)
cnv5 = keras.layers.Dropout(0.2)(cnv5)
cnv5 = keras.layers.Conv2D(512, 3, activation = 'relu', padding='same')(cnv5)



upcnv1 = keras.layers.Conv2DTranspose(256, 2,strides=2)(cnv5)
upcnv1 = keras.layers.Concatenate()([upcnv1,cnv4])
cnv6 = keras.layers.Conv2D(256, 3, activation = 'relu', padding='same')(upcnv1)
cnv6 = keras.layers.Dropout(0.2)(cnv6)
cnv6 = keras.layers.Conv2D(256, 3, activation = 'relu', padding='same')(cnv6)



upcnv2 = keras.layers.Conv2DTranspose(256, 2, padding= 'same',strides=2)(cnv6)
upcnv2 = keras.layers.Concatenate()([upcnv2,cnv3])
cnv7 = keras.layers.Conv2D(128, 3, activation = 'relu', padding='same')(upcnv2)
cnv7 = keras.layers.Dropout(0.2)(cnv7)
cnv7 = keras.layers.Conv2D(128, 3, activation = 'relu', padding='same')(cnv7)


upcnv3 = keras.layers.Conv2DTranspose(256, 2, padding= 'same',strides=2)(cnv7)
upcnv3 = keras.layers.Concatenate()([upcnv3,cnv2])
cnv8 = keras.layers.Conv2D(64, 3, activation = 'relu', padding='same')(upcnv3)
cnv8 = keras.layers.Dropout(0.2)(cnv8)
cnv8 = keras.layers.Conv2D(64, 3, activation = 'relu', padding='same')(cnv8)


upcnv4 = keras.layers.Conv2DTranspose(256, 2, padding= 'same',strides=2)(cnv8)
upcnv4 = keras.layers.Concatenate()([upcnv4,cnv1])
cnv9 = keras.layers.Conv2D(32, 3, activation = 'relu', padding='same')(upcnv4)
cnv9 = keras.layers.Dropout(0.2)(cnv9)
cnv9 = keras.layers.Conv2D(32, 3, activation = 'relu', padding='same')(cnv9)


output = keras.layers.Conv2D(4,1, activation = 'softmax', padding='same')(cnv9)


model = keras.Model(inputs= inputs,outputs = output)

# Defining the loss and optimizer for compilation

loss = sm.losses.DiceLoss(class_weights= class_weights)
metrics = sm.metrics.IOUScore(class_weights= class_weights)
model.compile(optimizer= keras.optimizers.Adagrad(), loss=loss, metrics=[metrics], )



# Visualizing the model

keras.utils.plot_model(model, show_shapes=True)



# Defining the callbacks for controlled training and avoiding overfitting

earlystop = keras.callbacks.EarlyStopping(patience=4, restore_best_weights= True)
checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(ROOT, 'model_weights', 'model_weights.h5'), save_best_only= True, save_weights_only= True)
csv_logger = keras.callbacks.CSVLogger(os.path.join(ROOT, 'epoch_results.csv'), append= True)



# Training process

history = model.fit(X_train,y_train_cat,batch_size=32,epochs = 100,callbacks=[earlystop, checkpoint, csv_logger],validation_split=0.2)



# Evaluating the model on test set

model.evaluate(X_test,y_test_cat)



from IPython.core.pylabtools import figsize
# Plotting the loss and IOU score 

df = pd.DataFrame(history.history)
df.plot(figsize = (10,8), grid = True)



# Predicing the masks for test images

y_preds = model.predict(X_test)
y_pred = np.argmax(y_preds,axis=3)



# Coparing the ground truth and prediciton along with images

number = np.random.randint(0,len(X_test))
img = X_test[number]
truth = y_test[number]
prediction = y_pred[number]

plt.figure(figsize=(16,8))
plt.subplot(131)
plt.title('Actual image')
plt.axis('off')
plt.imshow(img[:,:,0],cmap = 'gray')
plt.subplot(132)
plt.title('Ground Truth')
plt.axis('off')
plt.imshow(truth[:,:],cmap = 'jet')
plt.subplot(133)
plt.title('Predicted Mask')
plt.axis('off')
plt.imshow(prediction,cmap = 'jet')
plt.tight_layout()
plt.show()

