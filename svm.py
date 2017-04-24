import os
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

from dataset_parser import DataSet


def load_or_train(filename, model, train_X, train_y, compress=3):
    if os.path.exists(filename):
        model = joblib.load(filename)
        print('load job from disk ' + filename + '...')
    else:
        print('starting new job...')
        model.fit(train_X, train_y)
        print('job done...')
        joblib.dump(model, filename, compress=compress)
        print('job persistence finished...')
    return model


def test_linear_svm(train_X, train_y):
    classifier = LinearSVC()
    return load_or_train('SVM.pkl', classifier, train_X, train_y)


def test_svm_cross_validation(train_X, train_y):
    classifier = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 5, 10, 100, 1000], 'gamma': [0.001, 0.005, 0.0001]}
    grid_search = GridSearchCV(classifier, param_grid, n_jobs=4, verbose=3)
    grid_search = load_or_train('grid_best_parameter.pkl', grid_search, train_X, train_y, compress=1)
    best_parameters = grid_search.best_estimator_.get_params()
    # print(best_parameters)
    classifier = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    return load_or_train('SVM_CV.pkl', classifier, train_X, train_y)


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
