# Kaggle-Data-Science-LungCancer
This is the repository of the EC500 C1 class project

The Mask.py creates the mask for the nodules inside a image 

The lung.py generates the training and testing data sets, which would be ready to feed into the the U-net.py to train with.

U-net.py trains the data with U-net structure CNN, and gives out the result

The data sets can be accessed from https://drive.google.com/drive/u/1/folders/0Bz-jINrxV740SWE1UjR4RXFKRm8

To run this, you would need the tensorflow installed, and some sevral other python packages, but all are easy to install

Download the dataset, and change the directory for this to run properly

To run the program, the following must be installed:

Tensorflow with the CPU
skimage
matlablib
numpy
SimpleITK
csv

lung.py --- Mask.py ---U-net.py

Moreover, for guys who want to futher explore Kaggle Data Science Super Bowl, we upload the code of the 2nd position of this competition(we already re-implement it), it's a great resource for us who are interested in machine learning to explore and lear. Please follow the instruction in README file inside 3D CONV folder.
