# -*- coding: utf-8 -*-
"""
Created on 03/04 2024

"""
import warnings
warnings.filterwarnings('ignore')

from keras.applications.vgg16 import VGG16#, Xception

#from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input

#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

img_size = 224 
num_classes = 2
BS = 10 #Batch Size
epochs = 10

train_datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode='nearest',
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
            'PetImages/train',
            target_size=(img_size, img_size), 
            batch_size=BS,
            class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
            'PetImages/test/',
            target_size=(img_size, img_size),
            batch_size=BS,
            class_mode='categorical',
            shuffle = False)

n_steps = train_generator.samples // BS
n_test_steps = test_generator.samples // BS

model = Model()
#model=Sequential()
'''
conv2d=Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=(img_size,img_size,3),padding='same', strides=1)
model.add(Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=(img_size,img_size,3),padding='same', strides=1))
model.add(MaxPool2D(2,2))
#model.add(Dropout(0.1))

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', strides=1))
model.add(MaxPool2D(2,2))
#model.add(Dropout(0.1))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', strides=1))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
'''
input_layer = Input(shape=(img_size, img_size, 3))

# Define the model layers
conv1 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', strides=1)(input_layer)
pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', strides=1)(pool1)
pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', strides=1)(pool2)
pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
dropout1 = Dropout(0.3)(pool3)

flat = Flatten()(dropout1)
dense1 = Dense(64, activation='relu')(flat)
dropout2 = Dropout(0.3)(dense1)
output_layer = Dense(num_classes, activation='softmax')(dropout2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Print the model summary
model.summary()
'========= 績效 =========='
opt = optimizers.Adam(learning_rate=0.0001)
#opt = optimizers.SGD(learning_rate=0.00001, momentum=0.9)
#opt = optimizers.RMSprop(learning_rate=0.00001, momentum=0.0, rho=0.9, epsilon=1e-07)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])         

history = model.fit(train_generator,
                              steps_per_epoch=n_steps, # 相當於擴增次數，n*(train size/batch size)，n is any positive integer
                              epochs=epochs,
                              validation_data=test_generator, 
                              validation_steps=n_test_steps)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

predicted_classes = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predicted_classes,axis=1)
test_class=test_generator.classes
score = accuracy_score(test_class, predicted_classes)
print("Accuracy: %.2f%%" % (score*100))
correct = np.where(predicted_classes==test_class)[0]
print ("Found %d correct labels" % len(correct))
incorrect = np.where(predicted_classes!=test_class)[0]
print ("Found %d incorrect labels" % len(incorrect))

print (confusion_matrix(test_class, predicted_classes))
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_class, predicted_classes, target_names=target_names))


# -*- coding: utf-8 -*-
"""
Created on 03/04 2024
"""
'''
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

img_size = 224
num_classes = 2
BS = 10  # Batch Size
epochs = 10

train_datagen = ImageDataGenerator(
     rotation_range=5,
     width_shift_range=0.1,
     height_shift_range=0.1,
     fill_mode='nearest',
     shear_range=0.2,
     zoom_range=0.2,
     horizontal_flip=True,
     vertical_flip=True,
     rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    'PetImages/train',
    target_size=(img_size, img_size),
    batch_size=BS,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    'PetImages/test/',
    target_size=(img_size, img_size),
    batch_size=BS,
    class_mode='categorical',
    shuffle=False)

n_steps = train_generator.samples // BS
n_test_steps = test_generator.samples // BS

# Define the input tensor
input_layer = Input(shape=(img_size, img_size, 3))

# Define the model layers
conv1 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', strides=1)(input_layer)
pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', strides=1)(pool1)
pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', strides=1)(pool2)
pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
dropout1 = Dropout(0.3)(pool3)

flat = Flatten()(dropout1)
dense1 = Dense(64, activation='relu')(flat)
dropout2 = Dropout(0.3)(dense1)
output_layer = Dense(num_classes, activation='softmax')(dropout2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Print the model summary
model.summary()

'========= 績效 =========='
opt = optimizers.Adam(learning_rate=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

history = model.fit(train_generator,
                    steps_per_epoch=n_steps,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=n_test_steps)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))
plt.plot(epochs_range, acc, 'bo', label='Training acc')
plt.plot(epochs_range, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs_range, loss, 'bo', label='Training loss')
plt.plot(epochs_range, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

predicted_classes = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predicted_classes, axis=1)
test_class = test_generator.classes
score = accuracy_score(test_class, predicted_classes)
print("Accuracy: %.2f%%" % (score * 100))
correct = np.where(predicted_classes == test_class)[0]
print("Found %d correct labels" % len(correct))
incorrect = np.where(predicted_classes != test_class)[0]
print("Found %d incorrect labels" % len(incorrect))

print(confusion_matrix(test_class, predicted_classes))
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(test_class, predicted_classes, target_names=target_names))
'''
