# -*- coding: utf-8 -*-
"""
Created on 03/04 2024

"""
'''
from keras.applications.vgg16 import VGG16#, Xception

#from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten

from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

img_size = 224 
num_classes = 2
BS = 8
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
            'PetImages/train/',
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

'========= Transfer Learning (VGG16) =========='
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(img_size, img_size, 3)) # at least 32x32
conv_base.trainable = True
#conv_base.summary()

model = Model()
#model=Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

'========= 績效 =========='
opt = optimizers.Adam(learning_rate=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])         
history = model.fit_generator(train_generator,
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
'''

# -*- coding: utf-8 -*-
"""
Created on 03/04 2024
"""
import warnings
warnings.filterwarnings('ignore')

from keras.applications.vgg16 import VGG16  # , Xception
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

img_size = 224
num_classes = 2
BS = 8
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
    'PetImages/train/',
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

'========= Transfer Learning (VGG16) =========='
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(img_size, img_size, 3))
conv_base.trainable = True

input_layer = Input(shape=(img_size, img_size, 3))
x = conv_base(input_layer)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

'========= 績效 =========='
opt = optimizers.Adam(learning_rate=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

history = model.fit(
    train_generator,
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
