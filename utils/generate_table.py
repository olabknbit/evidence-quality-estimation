from typing import List

import numpy as np

from utils.metrics import read_test_result_from_metrics_file


def generate_table_from_filenames(filenames: List[str], accuracies: List[float]) -> None:
    from tabulate import tabulate

    all_features = set()
    for features in filenames:
        for feature_name in features.split('__'):
            all_features.add(feature_name)
    all_features = sorted(list(all_features))

    lines = []
    for i, (filename, acc) in enumerate(zip(filenames, accuracies)):
        features = filename.split('__')
        line = ['X' if f in features else '.' for f in all_features]
        line.append(acc)
        lines.append(line)
    all_features.append('acc')
    all_features_as_ids = [str(i) for i in range(len(all_features))]
    lines.sort(key=lambda x: x[-1])
    for i, feature in enumerate(all_features):
        print(i, '\t', feature)
    print(tabulate(lines, headers=all_features_as_ids))


def generate_accuracies_table_from_filenames(filenames: List[str], base_features_modes: List[str],
                                             accuracies: List[float]) -> None:
    from tabulate import tabulate

    methods = sorted(list(set(base_features_modes)))
    feature_names = sorted(list(set(filenames)))

    lines = []
    for _ in feature_names:
        line = ["fn"]
        for _ in methods:
            line.append(0.)
        lines.append(line)

    means = {method: [] for method in methods}
    for feature_name, bfm, acc in zip(filenames, base_features_modes, accuracies):
        if acc and filenames != "publication_year":
            means[bfm].append(acc)

    means = {method: np.mean(l) for method, l in means.items()}
    means = [[k, v] for k, v in sorted(means.items(), key=lambda item: item[1])]

    for filename, bfm, acc in zip(filenames, base_features_modes, accuracies):
        feature_name = filename.replace("disease_or_syndrome", "dos")
        lines[feature_names.index(filename)][0] = feature_name
        lines[feature_names.index(filename)][methods.index(bfm) + 1] = acc
    for i, method in enumerate(methods):
        method = method.replace("oversampling", "ovs")
        method = method.replace("undersampling", "und")
        method = method.replace("sklearn", "skl")
        method = method.replace("lemmatizer", "lem")
        method = method.replace("master", "mas")
        method = method.replace("balanced", "bal")
        methods[i] = method

    print()
    print(tabulate(lines, headers=['feature_name'] + methods))
    print()
    print(tabulate(means, headers=['method', 'mean_accuracy']))
    print()


def read_filenames_and_accuracies_from_filepaths(filepaths: List[str], metric: str):
    filenames = []
    base_features_modes = []
    accuracies = []
    for path in filepaths:
        result = read_test_result_from_metrics_file(path=path)
        features_name = path.split('/')[-1][:-len('_metrics.txt')]
        base_features_mode = path.split('/')[-3]
        filenames.append(features_name)
        base_features_modes.append(base_features_mode)
        accuracies.append(getattr(result.metrics, metric))
    return filenames, base_features_modes, accuracies


def generate_accuracies_table(filepaths: List[str], metric="balanced_accuracy_adjusted"):
    filenames, base_features_modes, accuracies = read_filenames_and_accuracies_from_filepaths(filepaths, metric)
    generate_accuracies_table_from_filenames(filenames, base_features_modes, accuracies)


def generate_table(filepaths: List[str], metric="balanced_accuracy_adjusted"):
    filenames, _, accuracies = read_filenames_and_accuracies_from_filepaths(filepaths, metric)
    generate_table_from_filenames(filenames, accuracies)
