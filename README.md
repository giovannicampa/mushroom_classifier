# Mushroom classifier
This project is a transfer-learning based multiclass classifier for mushrooms.

The model used is **InceptionV3**, and the training data was downloaded from Google Images through the
[Download all images extension](https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en) Chrome extension.

## Data
The data belongs to n different mushroom classes, namely _boletus, cantharellus, amanita_ and _macrolepiota_.
Among the downloaded pictures, the ones showing groups of the same mushroom have been deleted. 

## Training
Since this a transfer learning operation, it can be divided into feature extraction and classification.
The feature extractor layers have been set as trainable, which improved overall accuracy but slowed down training.
The classification part of the net consists of two Dense layers of 512 and 128 neurons respectively.

The picture below shows the accuracy and loss of the training and test sets over the epochs. Thanks to the **early stopping** callback (patience of 3 epochs), the training was interrupted, thus preventing overfitting. The patience of 3 epochs let the training progress beyond the optimal point to check whether there could have been an improvement. The weights of the best epoch were then restored.



## Results
The model achieves a validation accuracy of 98.4%. To understand how this is made up, the confusion matrix seen below has been generated.

<figure>
  <img src="https://github.com/giovannicampa/mushroom_classifier/blob/master/pictures/confusion_matrix.png" width="700">
  <figcaption>Tensorboard log of the training session</figcaption>
</figure>


### Interpretability
In order to correctly interpret the results, a interpretability analysis has been made. Bot the **class activation map**, as the **saliency map** have been calculated for pictures of each class.

The results can be seen below. Blue areas indicate increased attention towards that feature (CAM) or pixel (saliency).

| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/giovannicampa/mushroom_classifier/blob/master/pictures/amanita.png">  |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/giovannicampa/mushroom_classifier/blob/master/pictures/boletus.png">|
|<img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/giovannicampa/mushroom_classifier/blob/master/pictures/cantharellus.png">  |  <img width="1604" alt="screen shot 2017-08-07 at 12 18 15 pm" src="https://github.com/giovannicampa/mushroom_classifier/blob/master/pictures/macrolepriota.png">|


