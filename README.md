# HDCR
Handwritten digital character recognition based on `scikit-learn` and MNIST database.

## Dataset Preprocessing

Firstly download the dataset from [this link](http://yann.lecun.com/exdb/mnist/).

`dataset_parser.py` uses `struct` to extract training and test data from downloaded binary files.

### Usage

- Initialize: `Dataset(features, labels)` where `features` and `labels` are the location of the features file and label file respectively.
- `get_images()` returns a `numpy` array containing original features of 28*28 dimensions.
- `get_hog_images()` returns a `numpy` array containing features decomposed by HOG (Histogram of Oriented Gradient) algorithm.
- `get_labels()` returns a `numpy` array containing labels.


- Additionally, `get_image_files` and `get_label_file` generate files containing images and labels in the current directory respectively.

## Training and Test

- `test_linear_svm(features, labels)` and `test_svm_cross_validation(features, labels)` use the linear SVM classifier (support vector machine) and the SVM classifier using `rbf` kernel function with parameter tuning by cross validation.
- `test(models, features, labels)` use one trained model to verify the model and get the correctness ratio.

