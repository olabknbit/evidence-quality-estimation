from typing import Any, Dict

from sklearn.model_selection import GridSearchCV

from base_methods.base_sklearn import get_pipeline, get_Xy
from ebm.dataset import Data
from ebm.dataset import get_train_dev_data, get_test_data
from params import BaseParams, SingleBaseParams, MultiBaseParams
from utils.feature_names import *

classifiers = {
    SVC_name: {
        'clf__C': [0.1, 1, 10, 100, 1000],
        'clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'clf__kernel': ['rbf', 'linear']},
}


def run_grid_search(train: Data, test: Data, params: BaseParams, parameters: Dict[str, Any]) -> None:
    X_train, y_train = get_Xy(train, params)
    X_test, y_test = get_Xy(test, params)

    clf = params.classifier

    pipeline = get_pipeline(params)
    grid = GridSearchCV(pipeline, param_grid=parameters, cv=10)
    grid.fit(X_train, y_train)
    y_predicted = grid.predict(X_test)

    print("best_score", grid.best_score_)

    index = grid.best_index_
    print(grid.cv_results_['mean_test_score'][index], grid.cv_results_['std_test_score'][index])

    from utils.metrics import calculate_metrics
    m = calculate_metrics(y_test, y_predicted)
    print(m.to_string())
    # print(feature_name)

    print("best_params", grid.best_params_)

    from utils.file_management import save_data_with_ultimate_dir_creation
    lines = ("clf: %s\n"
             "mean_test_score = %3.4f\n"
             "std_test_score %3.4f\n"
             "metrics:\n%s\n"
             "best_params: %s\n"
             "------------------\n" %
             (clf,
              grid.cv_results_['mean_test_score'][index] * 100,
              grid.cv_results_['std_test_score'][index] * 100,
              m.to_string(),
              str(grid.best_params_)))
    path = "out/" + params.dirname + "_hyperparams/" + params.get_features_names_str() + '.txt'
    save_data_with_ultimate_dir_creation(path, [lines])


def single_features():
    train = get_train_dev_data()
    test = get_test_data()

    for classifier in classifiers.keys():
        for feature in all_feature_names:
            print(feature)
            params = SingleBaseParams(feature_name=feature, classifier=classifier, dirname="new", params={}, pca=False)
            run_grid_search(train=train, test=test, params=params, parameters=classifiers[classifier])


def multiple_features():
    train = get_train_dev_data()
    test = get_test_data()

    # all_subsets = get_all_feature_subsets()
    all_subsets = [[publication_type_key, publication_year_key]]
    for classifier in classifiers.keys():
        for subset in all_subsets:
            print(subset)
            params = MultiBaseParams(features=subset, classifier=classifier, dirname="new", params={}, pca=False)
            run_grid_search(train, test, params, parameters=classifiers[classifier])


if __name__ == "__main__":
    pass
