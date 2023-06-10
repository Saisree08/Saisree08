#!/usr/bin/env python
# coding: utf-8

# In[ ]:


data_directory = "data/ravdess"


# In[ ]:


root_path = '/'


# In[ ]:


import os
os.chdir(root_path)


# In[ ]:


get_ipython().system('pip install PyDrive')

from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


# In[ ]:


auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


# In[ ]:


if not os.path.exists('data'): 
    os.makedirs('data')

downloaded = drive.CreateFile({'id':"1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7"})   # replace the id with id of file you want to access
downloaded.GetContentFile('speech-emotion-recognition-dataset.zip')

get_ipython().system('mv speech-emotion-recognition-dataset.zip data')

if not os.path.exists(data_directory): 
    os.makedirs(data_directory)

get_ipython().system('unzip data/speech-emotion-recognition-dataset.zip -d datasets/ravdess')


# In[ ]:


import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, f1_score
import pickle
import keras
from keras import layers, Sequential
from keras.layers import Conv1D, Activation, Dropout, Dense, Flatten, MaxPooling1D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import sklearn.metrics as metrics


# In[ ]:


emotions = {
    '01':'neutral',
    '02':'calm',
    '03':'happy',
    '04':'sad',
    '05':'angry',
    '06':'fearful',
    '07':'disgust',
    '08':'surprised'
}


# In[ ]:


def extract_feature(data, sr, mfcc, chroma, mel):    
    if chroma:                          
        stft = np.abs(librosa.stft(data))  
    result = np.array([])
    if mfcc:                          
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:                          
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:                             
        mel = np.mean(librosa.feature.melspectrogram(data, sr=sr).T,axis=0)
        result = np.hstack((result, mel))
        
    return result 


# In[ ]:


def noise(data, noise_factor):
    noise = np.random.randn(len(data)) 
    augmented_data = data + noise_factor * noise
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


# In[ ]:


def shift(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
        
    return augmented_data


# In[ ]:


def load_data(save=False):
    x, y = [], []
    for file in glob.glob(data_directory + "/Actor_*/*.wav"): 
        data, sr = librosa.load(file)
        feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]  
        y.append(emotion)
        n_data = noise(data, 0.001)
        n_feature = extract_feature(n_data, sr, mfcc=True, chroma=True, mel=True)
        x.append(n_feature)
        y.append(emotion)
        s_data = shift(data,sr,0.25,'right')
        s_feature = extract_feature(s_data, sr, mfcc=True, chroma=True, mel=True)
        x.append(s_feature)
        y.append(emotion)
    if save==True:
        np.save('X', np.array(x))
        np.save('y', y)
    return np.array(x), y


# In[ ]:


X, y = load_data(save=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=9)


# In[ ]:


labelencoder = LabelEncoder()
labelencoder.fit(y_train)
le_name_mapping = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
print(le_name_mapping)
y_train = labelencoder.transform(y_train)
y_test = labelencoder.transform(y_test)


# In[ ]:


print(f'Features extracted: {x_train.shape[1]}')


# In[ ]:


model = Sequential()
model.add(Conv1D(256, 5,padding='same', input_shape=(180,1))) 
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))) 
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))) 
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))) 
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=8,
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-5)
                )
) 
model.add(Activation('softmax'))
opt = keras.optimizers.adam(decay=1e-6)


# In[ ]:


XProccessed = np.expand_dims(x_train, axis=2)
XTestProcessed = np.expand_dims(x_test, axis=2)
history = model.fit(XProccessed, y_train, epochs=100, validation_data=(XTestProcessed, y_test), batch_size=64)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


# In[ ]:


y_pred = model.predict(XTestProcessed)


# In[ ]:


f1_score(y_test,np.argmax(y_pred,axis=-1),average='weighted')


# In[ ]:


model.summary()


# In[ ]:


if not os.path.exists('models'): 
    os.makedirs('models')

model.save("models/model3.h5")


# In[ ]:


loaded_model = keras.models.load_model("models/model3.h5")


# In[ ]:


def load_single_data(file):
    x, y = [], []
    file_name = os.path.basename(file)
    emotion = emotions[file_name.split("-")[2]]
    data, sr = librosa.load(file)
    feature = extract_feature(data, sr, mfcc=True, chroma=True, mel=True)
    x.append(feature)
    y.append(emotion)
    return np.array(x), y


# In[ ]:


XX, yy = load_single_data("data/ravdess/Actor_01/03-01-05-02-02-02-01.wav")


# In[ ]:


yy


# In[ ]:


XXTemp=np.expand_dims(XX, axis=2)
XX, yy = load_single_data("data/ravdess/Actor_01/03-01-05-02-02-02-01.wav") = model.predict(XXTemp)


# In[ ]:


list(y_pred)


# In[ ]:




