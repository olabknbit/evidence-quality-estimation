import os.path
from typing import List, Tuple, Optional

from params import deserialize, BaseParams
from utils.file_management import get_all_dirpaths


def get_all_existing_base_features(single: bool, multiple: bool) -> List[BaseParams]:
    single_base_features_dirs = ["sklearn", "sklearn_master_pt"]

    multiple_base_features_dirs = ["sklearn_mixed", "sklearn_master", "mixed"]

    all_params = []

    for dirname in single_base_features_dirs + multiple_base_features_dirs:
        dir_with_features = os.path.join("data", dirname, "test")
        feature_paths = get_all_dirpaths(dir_with_features)
        for feature_path in feature_paths:
            feature_name = feature_path.split('/')[-1]
            clf_paths = get_all_dirpaths(feature_path)
            for clf_path in clf_paths:
                classifier = clf_path.split('/')[-1]
                params = deserialize(feature_name, dirname, classifier)
                if len(params.features) == 1 and single:
                    all_params.append(params)
                elif len(params.features) > 1 and multiple:
                    all_params.append(params)
    print(len(all_params))
    return all_params


def filter_feature_sets_for_ensemble_with_highest_accuracies(feature_sets: List[BaseParams], acc: float = 0.6) -> List[
    BaseParams]:
    from features.base_features import read_one_feature
    from utils.metrics import calculate_metrics
    from utils.mapping import proba_to_letters

    def read(feature: BaseParams) -> Tuple[Optional[BaseParams], float]:
        try:
            _, x_train, y_train = read_one_feature(feature, train_vs_test="train", only_ab=False, to_binary=False)
            _, x_test, y_test = read_one_feature(feature, train_vs_test="test", only_ab=False, to_binary=False)
            x = list(x_train) + list(x_test)
            y = list(y_train) + list(y_test)

            metrics = calculate_metrics(y, proba_to_letters(x))
            print(feature.get_features_names_str(), metrics.accuracy)
            return (feature, metrics.accuracy)
        except Exception as ignored:
            return (None, 0.)

    all_names = {feature.get_features_names_str(): read(feature) for feature in feature_sets}
    all_fnames = list(
        filter(lambda x: x[0] is not None and x[1] > acc, reversed(sorted(all_names.values(), key=lambda x: x[1]))))
    print(len(all_fnames))
    return [x[0] for x in all_fnames]


def get_all_single_feature_subsets_for_ensemble() -> List[List[BaseParams]]:
    from utils.stacking import get_all_subsets

    all_subsets = get_all_subsets(get_all_existing_base_features(single=True, multiple=False))
    all_subsets = list(filter(lambda x: len(x) >= 2, all_subsets))
    print(all_subsets)
    return all_subsets


def get_all_highest_perf_single_feature_subsets_for_ensemble(acc=0.5) -> List[List[BaseParams]]:
    from utils.stacking import get_all_subsets

    features_with_highest_accuracies = filter_feature_sets_for_ensemble_with_highest_accuracies(
        feature_sets=get_all_existing_base_features(single=True, multiple=False), acc=acc)
    print(len(features_with_highest_accuracies))
    all_subsets = get_all_subsets(features_with_highest_accuracies)
    all_subsets = list(filter(lambda x: len(x) >= 2, all_subsets))
    print(all_subsets)
    return all_subsets


def get_all_highest_perf_multiple_feature_subsets_for_ensemble(acc=0.6) -> List[List[BaseParams]]:
    from utils.stacking import get_all_subsets

    features_with_highest_accuracies = filter_feature_sets_for_ensemble_with_highest_accuracies(
        feature_sets=get_all_existing_base_features(single=False, multiple=True), acc=acc)

    all_subsets = get_all_subsets(features_with_highest_accuracies)
    all_subsets = list(filter(lambda x: len(x) >= 2, all_subsets))
    import random
    random.shuffle(all_subsets)

    return all_subsets
