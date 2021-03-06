#! python3
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import urllib
import math
import datetime
from shutil import copyfile
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler

import pathlib


# tf.config.experimental.set_visible_devices([], 'GPU') # Enforcing the usage of the CPU
random.seed(42)

# ---------------------------------------------------------------------------------------------------------
# Prepare data

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[training_length:]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


mushroom_classes = ["amanita", "boletus", "cantharellus", "macrolepiota"] # "armillaria",
TRAINING_DIR = "./images/training/"
TESTING_DIR = "./images/testing/"


split_size = .7

if not(os.path.exists("./images/training/")):

    for mushroom in mushroom_classes:
        pathlib.Path(TRAINING_DIR + mushroom + "/").mkdir(parents=True, exist_ok=True)
        pathlib.Path(TESTING_DIR + mushroom + "/").mkdir(parents=True, exist_ok=True)

        SOURCE_DIR_MUSHROOM = "./images/" + mushroom + "/"
        TRAINING_DIR_MUSHROOM = TRAINING_DIR + mushroom + "/"
        TESTING_DIR_MUSHROOM = TESTING_DIR + mushroom + "/"
        split_data(SOURCE_DIR_MUSHROOM, TRAINING_DIR_MUSHROOM, TESTING_DIR_MUSHROOM, split_size)



# Experiment with your own parameters to reach 99.9% validation accuracy or better
train_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=20,
                                                    target_size=(299, 299),
                                                    class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(TESTING_DIR,
                                                              batch_size=20,
                                                              class_mode='categorical',
                                                              target_size=(299, 299))


# ---------------------------------------------------------------------------------------------------------------
# - Callbacks

# Early stopping callback
es_callback = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=0, restore_best_weights= True)


# Learning rate scheduler (not needed for RMSprop)
def generate_lr_scheduler(initial_learning_rate = 0.001):

    def lr_step_decay(epoch, lr):
        drop_rate = 0.5
        epochs_drop = 10.0
        return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))

    return LearningRateScheduler(lr_step_decay)

learning_rate = 0.0001
lr_callback = generate_lr_scheduler(initial_learning_rate= learning_rate)


# Tensorboard callback
current_path = os.getcwd()
now = datetime.datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S").strip(" ").replace("/", "_").replace(":", "_").replace(" ", "_")
log_directory = current_path + "/tensorboard/" + dt_string

tb_callback = TensorBoard(log_dir = log_directory, profile_batch=0, histogram_freq=1)



# ---------------------------------------------------------------------------------------------------------------
# Defined model
weights_file = "inception_v3.h5"

if not os.path.exists(weights_file):
    weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    urllib.request.urlretrieve(weights_url, weights_file)

# Instantiate the model
pre_trained_model = InceptionV3(input_shape=(299, 299, 3),
                                include_top=False,
                                weights=None)

# load pre-trained weights
pre_trained_model.load_weights(weights_file)

# Decide whether to train also the feature extraction
freeze_layers = False
if freeze_layers == True:
    for layer in pre_trained_model.layers:
        layer.trainable = False


last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Define model
x = tf.keras.layers.GlobalAveragePooling2D()(last_output)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dense(len(mushroom_classes), activation="softmax", name="classification")(x)

model = tf.keras.Model(inputs=pre_trained_model.input, outputs = x)
model.summary()

model.compile(optimizer=RMSprop(lr=learning_rate), 
                loss='categorical_crossentropy',
                metrics = ['acc'])


EPOCHS = 10

history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=EPOCHS,
            verbose=1,
            callbacks= [tb_callback, es_callback])


model.save('saved_model/my_model' + dt_string)


# ---------------------------------------------------------------------------------------------------------------
# Model evaluation

fig, ax = plt.subplots()

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        TESTING_DIR,
        shuffle=False,
        class_mode='categorical')

Y_pred = model.predict_generator(test_generator, steps=len(test_generator))
y_pred = np.argmax(Y_pred, axis=1)


confusion_mat = confusion_matrix(y_true = test_generator.classes,
                                y_pred = y_pred,
                                normalize="true")

im = ax.imshow(confusion_mat)


ax.set_xticklabels(mushroom_classes)
ax.set_yticklabels(mushroom_classes)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

for row in range(confusion_mat.shape[0]):
    for col in range(confusion_mat.shape[1]):
        text = ax.text(
                x = col,
                y = row,
                s = f"{confusion_mat[row, col]:.2f}",
                ha="center",
                va="center",
                color="w")


ax.set_xticks(list(range(len(mushroom_classes))))
ax.set_yticks(list(range(len(mushroom_classes))))

ax.set_xlabel(f"Predicted label", fontsize = 18)
ax.set_ylabel(f"True label", fontsize = 18)

fig.tight_layout()
plt.show()