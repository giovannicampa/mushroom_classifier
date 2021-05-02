#! /usr/bin/python3
import os

import tensorflow as tf

import keras
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,GlobalAveragePooling2D
from keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import cv2


# Load model from savel_model directory
save_models_list = os.listdir("saved_model/")

saved_model = save_models_list[1]
model = tf.keras.models.load_model("saved_model/" + saved_model)
model.summary()

batch_size_input, height_input, width_input, colors_input = model.input_shape


# gets weights connecting the GAP to the Dense layer
# [0-> weights, 1-> bias]
gap_weights = model.layers[-3].get_weights()[0]

# Defining class activation map model with modified outputs [features, result]
cam_model = Model(inputs = model.inputs, outputs = [model.layers[-6].output, model.layers[-1].output])

classes = ["amanita", "boletus", "cantharellus", "macrolepiota"]

def generate_cam(image_value, features, results):
    """ Displays the class activation map of an image

    Args:
    image_value (tensor) -- preprocessed input image with size (w = 150, l = 150, color = 3)
    features (array) -- features of the image extracted by the last conv layer, shape (batch = 1, w = 37, h = 37, filters = 128)
    results (array) -- output of the softmax layer, shape (len(classes))
    """

    # there is only one image in the batch so we index at `0`
    features_for_img = features[0]
    prediction = results[0]

    # there is only one unit in the output so we get the weights connected to it
    class_activation_weights = gap_weights[:,0]

    # upsample to the image size (37 = dimension of the output from the last conv2d)
    class_activation_features = sp.ndimage.zoom(features_for_img, (height_input/7, width_input/7, 1), order=2)

    # compute the intensity of each feature in the CAM
    cam_output  = np.dot(class_activation_features,class_activation_weights)

    return results, cam_output


def generate_saliency_map(image_value, class_true_idx):
    """ Displays the saliency map of an image

    Args:
    image_value (tensor) -- preprocessed input image with size (w = 150, l = 150, color = 3)
    class_true_idx (scalar) -- index of the real class for a one hot encoding representation
    """

    # number of classes in the model's training data
    num_classes = len(classes)

    # convert to one hot representation to match our softmax activation in the model definition
    expected_output = tf.one_hot([class_true_idx] * image_value.shape[0], num_classes)

    with tf.GradientTape() as tape:
        # cast image to float
        inputs = tf.cast(image_value, tf.float32)

        # watch the input pixels
        tape.watch(inputs)

        # generate the predictions
        predictions = model(inputs)

        # get the loss
        loss = tf.keras.losses.categorical_crossentropy(expected_output, predictions)

    # get the gradient with respect to the inputs
    gradients = tape.gradient(loss, inputs)


    # reduce the RGB image to grayscale
    grayscale_tensor = tf.reduce_sum(tf.abs(gradients), axis=-1)

    # normalize the pixel values to be in the range [0, 255].
    # the max value in the grayscale tensor will be pushed to 255.
    # the min value will be pushed to 0.
    normalized_tensor = tf.cast(
        255
        * (grayscale_tensor - tf.reduce_min(grayscale_tensor))
        / (tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor)),
        tf.uint8,
    )

    # remove the channel dimension to make the tensor a 2d tensor
    normalized_tensor = tf.squeeze(normalized_tensor)

    gradient_color = cv2.applyColorMap(normalized_tensor.numpy(), cv2.COLORMAP_HOT)
    gradient_color = gradient_color / 255.0
    super_imposed = cv2.addWeighted(image_value[0], 0.5, gradient_color, 0.5, 0.0)

    return super_imposed


# utility function to preprocess an image and show the CAM
def convert_and_classify(image, class_true_idx):
    """ Show the interpretabilty of the image

    Plot the class activation and the saliency maps
    """
    # load the image
    img = cv2.imread(image)

    # preprocess the image before feeding it to the model
    img = cv2.resize(img, (height_input, width_input)) / 255.0

    # add a batch dimension because the model expects it
    tensor_image = np.expand_dims(img, axis=0)

    # get the features and prediction
    features, results = cam_model.predict(tensor_image)

    # Generate the interpretability maps
    results, cam_output = generate_cam(tensor_image, features, results)
    super_imposed = generate_saliency_map(tensor_image, class_true_idx = 0)

    # visualize the results
    print(f'sigmoid output: {results}')

    fig, ax = plt.subplots(1,2)
    fig.suptitle(f"Classified as {classes[np.argmax(results[0])]}", fontsize = 17)
    ax[0].imshow(cam_output, cmap='jet', alpha=0.5)
    ax[0].imshow(tf.squeeze(tensor_image), alpha=0.5)
    ax[0].set_title("Class activation map", fontsize = 15)
    ax[1].imshow(super_imposed)
    ax[1].set_title("Saliency map", fontsize = 15)
    plt.show()


# Plot a series of pictures for each class
for i, mushroom in enumerate(classes):

    path_2_picture = "./images/" + mushroom + "/"
    picture_list = os.listdir(path_2_picture)

    for j in range(10):
        convert_and_classify(path_2_picture + picture_list[j], class_true_idx = 0)



