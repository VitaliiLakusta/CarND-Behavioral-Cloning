#%%
import keras
from scipy import ndimage
import matplotlib.pyplot as plt
import csv
import numpy as np

# input img size 160(y)x320(x)

# DONE Crop images on top and bottom 
# DONE Convert model to nvidia model
# TODO: convert data fetching for training into generator function
# TODO Apply angle correction for left and right images
# TODO: flip images along horizontal axis as well -> more training data
# TODO: If necessary, train with your own collected driving data, tune the model.


# %%
img = ndimage.imread('./data/IMG/center_2016_12_01_13_30_48_287.jpg')
# crop image same way as used in model to see if right amount is cropped
img = img[70:-25, :]
plt.imshow(img)

#%% 
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile, )
    for line in reader:
        lines.append(line)
lines = lines[1:]
images = []
measurements = []
for line in lines: 
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    img = ndimage.imread(current_path)
    images.append(img)
    steering_angle = float(line[3])
    measurements.append(steering_angle)

X_train = np.array(images)
y_train = np.array(measurements)

print(y_train[50])
plt.imshow(X_train[50])

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

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose=1)

model.save('model.h5')
model.save_weights('model_weights.h5')