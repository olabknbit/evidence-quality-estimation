from typing import List

from features.base_features import read
from params import BaseParams
from utils.stacking import pca_ensemble_features


def run(features: List[BaseParams], classifier_name: str, to_binary=False,
        use_only_ab=False, balanced=False):
    _, X_train, y_train = read(features, train_vs_test='train', only_ab=use_only_ab, to_binary=to_binary)
    _, X_test, y_test = read(features, 'test', only_ab=use_only_ab, to_binary=to_binary)

    if use_only_ab:
        X_train, X_test = pca_ensemble_features(X_train, X_test)

    from utils.mapping import letters_to_ints
    y_train = letters_to_ints(y_train)
    y_test = letters_to_ints(y_test)

    from classifier.models import get
    from sklearn.model_selection import GridSearchCV
    from utils.metrics import confidence_intervals
    from utils.stacking import replace_fn
    import numpy as np
    params = {}
    if balanced:
        params = {'class_weight': 'balanced'}
    model = get(classifier_name, params=params)
    grid = GridSearchCV(model, param_grid={}, cv=10)
    grid.fit(X_train, y_train)
    accuracies = []
    for i in range(10):
        model.fit(X_train, y_train)
        mean_score = model.score(X_test, y_test)
        accuracies.append(mean_score * 100)
    l, r = confidence_intervals(accuracies)
    mea = float(np.mean([l, r]))

    print("best_score", grid.best_score_)

    index = grid.best_index_
    print(grid.cv_results_['mean_test_score'][index], grid.cv_results_['std_test_score'][index])

    print("best_params", grid.best_params_)
    feature_names_str = ', '.join([replace_fn(fn.get_features_names_str()) for fn in features])

    with open("out/sklearn_mixed_hyperparams.txt", 'a') as f:
        f.write("clf: %s\n"
                "mean_test_score = %3.4f\n"
                "std_test_score %3.4f\n"
                "metrics:\n"
                "accuracy (A): %3.2f\n"
                "best_params: %s\n"
                "feature_names: %s\n"
                "%s & %s & %s & %.2f & %.2f & %.2f & %.2f-%.2f \\\\\n"
                "------------------\n" %
                (classifier_name,
                 grid.cv_results_['mean_test_score'][index] * 100,
                 grid.cv_results_['std_test_score'][index] * 100,
                 mea,
                 str(grid.best_params_),
                 '\n'.join(feature_names_str),
                 feature_names_str, classifier_name, str(params), grid.cv_results_['mean_test_score'][index] * 100,
                 grid.cv_results_['std_test_score'][index] * 100, mea, l, r))

    return mea
