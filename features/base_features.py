from typing import Tuple, Dict

from ebm.filenames import get_features_dir
from utils.feature_names import *
from params import BaseParams


class FeaturesData:
    def __init__(self):
        self.evidence_ids, self.Xs, self.ys = [], [], []

    def append(self, evidence_id, x, y):
        self.evidence_ids.append(evidence_id)
        self.Xs.append(x)
        self.ys.append(y)

    def return_lists(self):
        return self.evidence_ids, self.Xs, self.ys

    def conv_Xs_to_label(self):
        letters = ['A', 'B', 'C']
        preds = []
        for xs in self.Xs:
            pred = max(range(len(xs)), key=xs.__getitem__)
            preds.append(letters[pred])
        return preds

    def count_accuracy(self):
        from utils.metrics import calculate_metrics
        return calculate_metrics(self.conv_Xs_to_label(), self.ys).accuracy


def pick_one_with_best_accuracy_score(classifiers_data: Dict[str, FeaturesData]):
    best_accuracy = 0.
    best_fd = None
    for fd in classifiers_data.values():
        acc = fd.count_accuracy()
        if acc > best_accuracy:
            best_accuracy = acc
            best_fd = fd
    return best_fd


def _read_features_file(dirpath: str, to_binary: bool = False) -> FeaturesData:
    from utils.file_management import get_all_filepaths_rec

    fd = FeaturesData()
    for filepath in get_all_filepaths_rec(dirpath):
        if filepath.split('/')[-1] == 'params':
            continue
        elems = filepath.split('/')
        evidence_id = elems[-1][:-len('.txt')]
        with open(filepath, 'r') as f:
            features = f.readlines()[0].split(' ')
            X = list(map(float, features))
            if to_binary:
                pred = max(range(len(X)), key=X.__getitem__)
                X = [0. for _ in range(3)]
                X[pred] = 1.
        y = elems[-2]

        fd.append(evidence_id, X, y)
    return fd


def read_one_feature(feature: BaseParams, train_vs_test: str, only_ab: bool, to_binary: bool) -> Tuple[
    List[str], List[List[float]], List[str]]:
    path = get_features_dir(feature.get_features_names_str(), feature.dirname, train_vs_test, feature.classifier)
    evidence_ids, Xs, ys = _read_features_file(path, to_binary=to_binary).return_lists()
    if only_ab:
        Xs = [[a, b] for a, b, c in Xs]

    return evidence_ids, Xs, ys


def read(features: List[BaseParams], train_vs_test: str, only_ab: bool, to_binary: bool) -> Tuple[
    List[str], List[List[float]], List[str]]:
    ev_ids = []
    combined_Xs = []
    trues = []
    for feature in features:
        evidence_ids, Xs, ys = read_one_feature(feature, train_vs_test=train_vs_test, only_ab=only_ab,
                                                to_binary=feature.to_binary)
        if not combined_Xs:
            combined_Xs = Xs
            ev_ids = evidence_ids
            trues = ys
        else:
            assert evidence_ids == ev_ids
            assert trues == ys
            for cxs, xs in zip(combined_Xs, Xs):
                cxs.extend(xs)
    return ev_ids, combined_Xs, trues
