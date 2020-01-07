# Arda Mavi
import os
import numpy as np
from os import listdir
from skimage.io import imread
from skimage.transform import resize
from keras.utils import to_categorical
#from sklearn.model_selection import train_test_split

# Settings:
num_class = 10


def get_img(data_path, img_size=64, grayscale_images=True):
    # Getting image array from path:
    img = imread(data_path, as_gray=grayscale_images)
    img = resize(img, (img_size, img_size, 1 if grayscale_images else 3))
    return img

def get_dataset(dataset_path='Dataset', img_size=64, as_gray=True):
    # Getting all data from data path:
    try:
        X = []
        Y = []
        if(as_gray):
            X = np.load('npy_dataset/X_grey.npy')
        else:
            X = np.load('npy_dataset/X_rgb.npy')
        Y = np.load('npy_dataset/Y_rgb.npy')
    except:
        labels = listdir(dataset_path) # Geting labels
        X = []
        Y = []
        for i, label in enumerate(labels):
            datas_path = dataset_path+'/'+label
            for data in listdir(datas_path):
                img = get_img(datas_path+'/'+data, grayscale_images=as_gray)
                X.append(img)
                Y.append(i)
        # Create dateset:
        X = np.array(X).astype('float32')/255.
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, num_class)
        if not os.path.exists('npy_dataset/'):
            os.makedirs('npy_dataset/')
        if(as_gray):
            np.save('npy_dataset/X_grey.npy', X)
        else:
            np.save('npy_dataset/X_rgb.npy', X)
        np.save('npy_dataset/Y_rgb.npy', Y)
    #X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X, Y
