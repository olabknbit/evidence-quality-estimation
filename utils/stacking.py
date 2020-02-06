from typing import Tuple

from utils.feature_names import *


def get_all_subsets(features: List):
    import itertools

    def findsubsets(s: List, n: int) -> List[Tuple]:
        return list(itertools.combinations(s, n))

    all_feature_sets = []
    for n in range(len(features)):
        feature_sets = findsubsets(features, n + 1)
        all_feature_sets.extend(feature_sets)
    return all_feature_sets


def pca_ensemble_features(x_train, x_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    from sklearn.decomposition import PCA
    pca = PCA(n_components='mle')
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    return x_train, x_test


def filter_filepaths_with_highest_accuracies(directories, n):
    from utils.generate_plots import filter_only_metrics_filenames
    from utils.file_management import get_all_filepaths_in_dirs
    from utils.metrics import read_test_result_from_metrics_file, Result, filter_filepaths_that_have_highest_accuracies
    filepaths = get_all_filepaths_in_dirs(directories)
    filepaths, _ = filter_filepaths_that_have_highest_accuracies(filepaths, n=n)
    filepaths = filter_only_metrics_filenames(filepaths)
    results: List[Result] = []
    for path in filepaths:
        result = read_test_result_from_metrics_file(path=path)
        results.append(result)
    results = list(sorted(results, key=lambda result: result.metrics.accuracy, reverse=True))
    return results


def get_feature_name_sets_with_highest_probable_accuracy(metrics_dirname):
    from ebm.filenames import get_metrics_dir
    from utils.file_management import get_all_filepaths_in_dirs
    from utils.metrics import filter_filepaths_that_have_highest_accuracies
    def filter_filepaths_with_highest_accuracies(directories):
        filepaths = get_all_filepaths_in_dirs(directories)
        filepaths, _ = filter_filepaths_that_have_highest_accuracies(filepaths, n=15)
        return filepaths

    directory = get_metrics_dir(metrics_dirname)

    filepaths = filter_filepaths_with_highest_accuracies([directory])
    from utils.generate_table import read_filenames_and_accuracies_from_filepaths
    filenames, _, accuracies = read_filenames_and_accuracies_from_filepaths(filepaths, metric="accuracy")
    fnames = list(reversed([fname.split('__') for fname in filenames]))
    return fnames


def replace_fn(feature_name) -> str:
    if feature_name == abstract_key:
        return abstract_key + " M(0)"
    if feature_name == conclusions_key:
        return conclusions_key + " M(0)"
    if feature_name == methods_key:
        return methods_key + " M(0)"
    if feature_name == title_key:
        return title_key + " M(0)"
    if feature_name == mesh_headings_key:
        return "MeSH terms"
    feature_name = feature_name.replace('_most_replaced', ' M(2)')
    feature_name = feature_name.replace('_disease_or_syndrome_replaced', ' M(1)')
    return feature_name.replace('_', ' ')
