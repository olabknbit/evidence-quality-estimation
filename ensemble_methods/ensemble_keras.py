from typing import List

import numpy as np
from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import to_categorical

from params import BaseParams
from utils.mapping import map_y_array_to_int
from utils.stacking import pca_ensemble_features


def get_model(Ni, optimizer):
    l = 512
    model = Sequential()
    model.add(Dense(l, activation='relu', input_shape=(Ni,)))
    model.add(Dense(l, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def run(features: List[BaseParams], optimizer, alpha=None, use_only_ab=False):
    from features.base_features import read
    _, x_train, y_train = read(features, train_vs_test="train", only_ab=use_only_ab, to_binary=False)
    _, x_test, y_test = read(features, train_vs_test="test", only_ab=use_only_ab, to_binary=False)
    x_train = np.array(x_train)
    x_test = np.array(x_test)

    if use_only_ab:
        x_train, x_test = pca_ensemble_features(x_train, x_test)

    y_train = np.array(map_y_array_to_int(y_train))
    y_test = np.array(map_y_array_to_int(y_test))
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    Ni = len(x_train[0])

    from sklearn.model_selection import KFold

    n_split = 10
    cvscores = []
    for train_index, test_index in KFold(n_split).split(x_train):
        X_train, X_test = x_train[train_index], x_train[test_index]
        Y_train, Y_test = y_train[train_index], y_train[test_index]

        model = get_model(Ni, optimizer)
        model.fit(X_train, Y_train, batch_size=100, epochs=10, verbose=0)

        scores = model.evaluate(X_test, Y_test, verbose=0)

        cvscores.append(scores[1])
    mean_test_score, std_test_score = np.mean(cvscores), np.std(cvscores)
    from utils.metrics import confidence_intervals

    accuracies = []
    for i in range(10):
        model = get_model(Ni, optimizer)
        model.fit(x_train, y_train, batch_size=100, epochs=10, verbose=0)
        score = model.evaluate(x_test, y_test)
        accuracies.append(score[1] * 100)
    l, r = confidence_intervals(accuracies)
    mea = float(np.mean([l, r]))

    # m = calculate_metrics(y_test, predicted)
    # method = keras_name + "_on_" + get_base_features_dirname()
    # r = Result(trainer=optimizer, method=method, metrics=m, features=feature_names, stacked=True, balanced=False,
    #            to_binary=False, use_only_ab=use_only_ab)
    # r.save_result()
    # print(m.to_string())
    # print(feature_name)

    from utils.stacking import replace_fn

    feature_names_str = ', '.join([replace_fn(fn.get_features_names_str()) for fn in features])
    print("%s & %s & %3.2f & %3.2f-%3.2f \\\\\n" % (feature_names_str, optimizer, mea, l, r))

    with open("out/keras_mixed_hyperparams.txt", 'a') as f:
        f.write("mean_test_score = %3.4f\n"
                "std_test_score %3.4f\n"
                "metrics:\n"
                "accuracy (A): %3.4f\n"
                "feature_names: %s\n"
                "%s & %s & 512, 512 & %.2f & %.2f & %.2f & %.2f-%.2f \\\\\n"
                "------------------\n" %
                (mean_test_score * 100,
                 std_test_score * 100,
                 mea,
                 '\n'.join(feature_names_str),
                 feature_names_str, optimizer, mean_test_score * 100, std_test_score * 100, mea, l, r))

    return mea
