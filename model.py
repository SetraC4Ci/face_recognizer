import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os

import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# tf.debugging.set_log_device_placement(True)

DATADIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DATADIR, "Images")
TRAIN_DATADIR = os.path.join(DATADIR, "train")
VAL_DATADIR = os.path.join(DATADIR, "test")

num_class = len(os.listdir(TRAIN_DATADIR))
IMG_SIZE = 120

model = Sequential()
model.add(ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False))
model.add(Flatten())
model.add(Dropout(.2))
model.add(Dense(num_class, activation='softmax'))


model.layers[0].trainable = False
model.summary()

model.compile(loss="categorical_crossentropy",
optimizer="adam",
metrics=['accuracy'])

train_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True, preprocessing_function=preprocess_input) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data


train_generator = train_image_generator.flow_from_directory(
    TRAIN_DATADIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_image_generator.flow_from_directory(
    VAL_DATADIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode='categorical'
)

print("NOW TRAINING...")
history = model.fit(
    x=train_generator,
    epochs=50,
    validation_data=validation_generator,
)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("TRAINING FINISHED, SAVING THE MODEL")
model.save("face_recognizer_model.h5")
