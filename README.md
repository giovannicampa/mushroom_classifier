# Mushroom classifier
This project is a transfer-learning based multiclass classifier for mushrooms.

The model used is **InceptionV3**, and the training data was downloaded from Google Images through the
[Download all images extension](https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en) Chrome extension.

## Data
The data belongs to n different mushroom classes, namely _boletus, cantharellus, amanita_ and _macrolepiota_.
Among the downloaded pictures, the ones showing groups of the same mushroom have been deleted. 

## Training
For the training the weights of the feature extractor part of the network have been frozen.
The classification part of the net consists of two Dense layers of 512 and 128 neurons respectively.

## Results
