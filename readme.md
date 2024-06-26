# DFO Bot (Big Otolith Tensor?)
Deep convolutional network for rapid automated otolith aging

## Directory structure:

Two main directories `preprocessing` and `model`.  Preprocessing is used to standardize and organize input images into a common format for the network, namely single otoliths cropped and then resized into a fixed resolution (e.g. 1000x1000).

Model contains the training and validation workflows for the ML model, as well as various helpers for running the scripts and performing the hyperparameter search.  

## Installation:
On Ubuntu need to install latest nvidia drivers (or whatever is suitable for the graphics card): `sudo apt install nvidia-driver-???` and a `pip install -r requirements.txt`


## References:
All the machine learning knowledge required is are covered in lectures 1-12 of this course:
https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/

This methods used in this code are not anything novel, they closely follow the aproach employed by Moen et al. 2018 and other recent otolith aging work:

Moen, E., et al. (2018). "Automatic interpretation of otoliths using deep learning." PLoS One 13(12): e0204713.

Sigurðardóttir, A. R., et al. (2023). "Otolith age determination with a simple computer vision based few-shot learning method." Ecological Informatics 76.

Politikos, D. V., et al. (2021). "Automating fish age estimation combining otolith images and deep learning: The role of multitask learning." Fisheries Research 242.

## Model: 

The current model structure consists of a pretrained version of ResNet50 with a single output instead of 1000 classes and uses an MSE loss.

Current data sets include American Plaice otoliths from the 2023 RV survey (~3500) and Herring otoliths from the 2019 season (4500).
Images containing an otolith pair are split into two seperate images resulting in a combined total training set of ~15000 images.
Basic image augmentation techniques (random rotation, cropping) and image normalization are implemented through PyTorch's dataloader class.  
The model trains on otoliths from both species simultaneously to maximize the overall generality of the model, best results so far have been in the ~65% accuracy range.  
