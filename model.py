#%%
import keras
from scipy import ndimage
import matplotlib.pyplot as plt
import csv
import numpy as np
import sklearn
import math
import random
import pickle

# %%
img = ndimage.imread('./data/IMG/center_2016_12_01_13_30_48_287.jpg')
plt.imshow(img)

# %%
# # crop image same way as used in model to see if right amount is cropped
plt.imshow(img[70:-25, :])

# %% 

path = '/opt/carnd_p3/data/'
# path = './data/'
#%% 
samples = []
with open(path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, skipinitialspace=True)
    for row in reader:
        samples.append(row)
samples = samples[1:]

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steering_angles = []
            for row in batch_samples: 
                steering_center = float(row[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                img_center = ndimage.imread(path + row[0])
                img_left = ndimage.imread(path + row[1])
                img_right = ndimage.imread(path + row[2])
                img_center_fl = np.fliplr(img_center)
                img_left_fl = np.fliplr(img_left)
                img_right_fl = np.fliplr(img_right)

                images.extend([img_center, img_center_fl, img_left, \
                    img_left_fl, img_right, img_right_fl])
                steering_angles.extend([steering_center, -steering_center, steering_left, \
                     -steering_left, steering_right, -steering_right])

            X_train = np.array(images)
            y_train = np.array(steering_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# %%
from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense, Cropping2D, Convolution2D

model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3), name='crop_1'))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, name='normalize_1'))
model.add(Convolution2D(24, 5, strides=(2,2), activation='relu', name='conv_1'))
model.add(Convolution2D(36, 5, strides=(2,2), activation='relu', name='conv_2'))
model.add(Convolution2D(48, 5, strides=(2,2), activation='relu', name='conv_3'))
model.add(Convolution2D(64, 3, activation='relu', name='conv_4'))
model.add(Convolution2D(64, 3, activation='relu', name='conv_5'))
model.add(Flatten(name='flatten_1'))
model.add(Dense(100, name="dense_1"))
model.add(Dense(50, name="dense_2"))
model.add(Dense(10, name="dense_3"))
model.add(Dense(1, name="dense_4"))

model.summary()

#%% 

# TRAIN AND SAVE THE MODEL

batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model.compile(loss='mse', optimizer='adam')
train_history = model.fit_generator(train_generator, \
    steps_per_epoch=math.ceil(len(train_samples)/batch_size), \
    validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/batch_size), \
    epochs=5, verbose=1)

model.save('model.h5')
model.save_weights('model_weights.h5')
with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(train_history.history, file_pi)

# %%
# Visualize loss

history = pickle.load(open('trainHistoryDict', 'rb'))

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# %%
