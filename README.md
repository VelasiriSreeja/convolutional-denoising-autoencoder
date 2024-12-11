# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

An autoencoder is an unsupervised neural network that compresses input images into lower-dimensional representations and then reconstructs them, aiming to produce outputs identical to the inputs. For training, we use the MNIST dataset, which contains 60,000 images of handwritten digits (each 28x28 pixels). The objective is to train a convolutional neural network on this dataset to classify each digit accurately into one of the 10 classes, ranging from 0 to 9.

## Convolution Autoencoder Network Model

![image](https://github.com/user-attachments/assets/0b6045b7-0086-44d5-a45e-af9ff9da6079)

## DESIGN STEPS

### STEP 1: 
Import the necessary libraries and dataset.

### STEP 2:
Load the dataset and scale the values for easier computation.

### STEP 3:
Add noise to the images randomly for both the train and test sets.

### STEP 4:
Build the Neural Model using,Convolutional,Pooling and Upsampling layers.

### STEP 5:
Pass test data for validating manually.

### STEP 6:
Plot the predictions for visualization.



## PROGRAM
### Name: JOYCE BEULAH R
### Register Number: 212222230058

### Program:
##### Importing necessary libraries
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, utils, models
from tensorflow.keras.datasets import mnist
```
##### Reading and Scaling the DataSet
```Python
(xtrain,_), (xtest,_) = mnist.load_data()
xtrain.shape
xtrainS = xtrain.astype('float32') / 255.
xtestS = xtest.astype('float32') / 255.
xtrainS = np.reshape(xtrainS,(len(xtrainS),28,28,1))
xtestS = np.reshape(xtestS,(len(xtestS),28,28,1))
```
##### Adding Noise to images
```Python
nf = 0.5
xtrainN=xtrainS+nf*np.random.normal(loc=0.0,scale=1.0,size=xtrainS.shape)
xtestN=xtestS+nf*np.random.normal(loc=0.0,scale=1.0,size=xtestS.shape)
xtrainN = np.clip(xtrainN, 0., 1.)
xtestN = np.clip(xtestN, 0., 1.)
```
##### Plotting Noisy images
```Python
n = 10
plt.figure(figsize=(20,2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(xtestN[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```
##### Developing Autoencoder DL Model
```Python
inp_img=keras.Input(shape=(28,28,1))
x=layers.Conv2D(16,(3,3),activation='relu',padding='same')(inp_img)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
encoder=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(encoder)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(3,3),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(3,3),activation='relu')(x)
x=layers.UpSampling2D((2,2))(x)
decoder=layers.Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)
model=keras.Model(inp_img,decoder)
model.summary()
```
##### Compiling and Fitting the Model
```Python
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(xtrainN,xtrainS,epochs=50,batch_size=128,shuffle=True,validation_data=(xtestN,xtestS))
```
##### Plotting the metrics
```Python
metrics=pd.DataFrame(model.history.history)
plt.figure(figsize=(7,2.5))
plt.plot(metrics['loss'], label='Training Loss')
plt.plot(metrics['val_loss'], label='Validation Loss')
plt.title('Training Loss vs. Validation Loss\n JOYCE BEULAH(212222230058)')
```
##### Predicting the model
```Python
decodeimg=model.predict(xtestN)
```
##### Plotting Original, Noisy, Reconstructed images
```Python
def display_images(xtestS, xtestN, decodeimg, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        for j, img in enumerate([xtestS,xtestN,decodeimg]):
            ax = plt.subplot(3, n, i + 1 + j * n)
            plt.imshow(img[i].reshape(28, 28), cmap='gray')
            ax.axis('off')
    plt.show()
display_images(xtestS, xtestN, decodeimg)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/676fc722-e29d-4a9a-ba5e-3a983964b1da)


### Original vs Noisy Vs Reconstructed Image

![image](https://github.com/user-attachments/assets/50b78446-4cc9-4077-8339-6d24f9cc8a60)

## Result:
Thus we have successfully developed a convolutional autoencoder for image denoising application.

