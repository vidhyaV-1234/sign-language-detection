import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Load the data
train = pd.read_csv("D:\project\sign-language-detection\sign_mnist_train.csv")
test = pd.read_csv("D:\project\sign-language-detection\sign_mnist_test.csv")
labels = train['label'].values


train.drop('label', axis=1, inplace=True)
images = train.values
images = np.array([np.reshape(i, (28, 28)) for i in images])

# Label binarizer
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(labels)

# Reshaping the images to add the channel dimension
images = images.reshape(images.shape[0], 28, 28, 1)

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)

# Normalizing the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Building the model
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(24, activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Training the model
history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# Saving the model
model.save("sign_minst_cnn.h5")

