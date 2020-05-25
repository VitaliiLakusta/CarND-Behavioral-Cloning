#%%
import keras
from scipy import ndimage
import matplotlib.pyplot as plt
import csv
import numpy as np

# input img size 160(y)x320(x)

# %%
img = ndimage.imread('./data/IMG/center_2016_12_01_13_30_48_287.jpg')
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
from keras.layers import Lambda, Flatten, Dense

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3), name='normalize_1'))
model.add(Flatten(name='flatten_1'))
model.add(Dense(1, name="dense_1"))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7, verbose=1)

model.save('model.h5')
model.save_weights('model_weights.h5')