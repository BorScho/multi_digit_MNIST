# multi_digit_MNIST
Construct a Neural Network (or a pipeline) to Read Multi Digit Handwritten Numbers composed form MNIST

The idea of this exercise is to generate a dataset of multi-digit numbers from images of single (handwritten) digits (Mnist) and to try to recognize these numbers.
We want to do this by cutting the multi-digit number into images of single digits and use an Mnist-Classificator on the single digits.
To this end we need a network to find the best points, where to cut a multi-digit number, then cut the image and present the single-digit images to another network, doing effectively an Mnist classification.

There are four Jupyter notebooks:
1. **"makeMultiDigits.ipynb"** - containing functions to glue multi-digit numbes from Mnist images. Images width is cut randomly at the front an back to make the resulting multi-digit image a little more realistic. Of course it will not be realistic because the digits are randomly chosen, i.e. the handwritting is from different people within one multi-digit number.
2. **"dataset_loaders_net.ipynb"** - containig training and test data-sets and -loaders and the definition for a net (**"MultiDigitMNISTNet"**) to single out "merge-points" i.e. the points on the horizontal axis where to cut the multi-digit number into single-digit numbers. 
3. **"single_mnist.ipynb"** - containing the definition of and train/evaluate functions for an Mnist-classifier. 
4. **"multidigit_end2end.ipynb"** - containing the code to cut multi-digit images according to the "merge-points" (should really have been named "cut-points"...) returned by the MultiDigitMNISTNet from 2. and perform a recognition, digit by digit, of the number. The entire pipeline is evaluated on the test set generated by the first notebook ("makeMultiDigits.ipynb"). Some simple probabilistic calculations show, what we should not expect to much of our algorithm.

Two more files are included: **MultiDigitMnist_training.py** and **MultiDigitMnist_functions.py** containing the same code as the notebooks. It seemed easier to import code into other notebooks from py-files then from other ipynb-files.
