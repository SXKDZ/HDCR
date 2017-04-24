import os
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

from dataset_parser import DataSet


def test_linear_svm(train_X, train_y):
    classifier = LinearSVC()
    if os.path.exists('SVM.model'):
        classifier = joblib.load('SVM.model')
        print('load model from disk SVM.model...')
    else:
        print('training model...')
        classifier.fit(train_X, train_y)
        print('training of %d samples done...' % len(train_X))
        joblib.dump(classifier, 'SVM.model', compress=3)
        print('model persistence finished...')
    return classifier


def test_svm_cross_validation(train_X, train_y):
    classifier = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(classifier, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_X, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    classifier = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    if os.path.exists('SVM_CV.model'):
        classifier = joblib.load('SVM_CV.model')
        print('load model from disk SVM_CV.model...')
    else:
        print('training model...')
        classifier.fit(train_X, train_y)
        print('training of %d samples done...' % len(train_X))
        joblib.dump(classifier, 'SVM_CV.model', compress=3)
        print('model persistence finished...')
    return classifier


def test(model, test_X, test_y):
    print('predicting samples...')
    predict = model.predict(test_X)
    print('prediction of %d samples done...' % len(predict))
    count = 0
    for i in range(len(test_X)):
        # print('%d %d' % (predict[i], test_y[i]))
        if predict[i] == test_y[i]:
            count += 1
    print(count / len(test_y))


if __name__ == '__main__':
    training_set = DataSet('train_images', 'train_labels')
    print('loading training set data...')
    train_X = training_set.get_hog_images()
    train_y = training_set.get_labels()
    print('loading training set data done...')

    testing_set = DataSet('test_images', 'test_labels')
    print('loading test set data...')
    test_X = testing_set.get_hog_images()
    test_y = testing_set.get_labels()
    print('loading test set data done...')

    print('model: SVM classifier')
    model_svm = test_linear_svm(train_X, train_y)
    test(model_svm, test_X, test_y)

    print('model: SVM classifier using cross validation')
    model_svm_cv = test_svm_cross_validation(train_X, train_y)
    test(model_svm_cv, test_X, test_y)
