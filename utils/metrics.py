from typing import List, Optional, Tuple

from params import Params


class Metrics:
    def __init__(self, accuracy, precision=None, average_precision=None, recall=None, jaccard=None, f1_score=None,
                 aed_score=None, balanced_accuracy=None, balanced_accuracy_adjusted=None, confusion_matrix=None,
                 confidence_intervals=None):
        self.accuracy: Optional[float] = accuracy
        self.balanced_accuracy: Optional[float] = balanced_accuracy
        self.balanced_accuracy_adjusted: Optional[float] = balanced_accuracy_adjusted
        self.precision: Optional[float] = precision
        self.average_precision: float = average_precision
        self.recall = recall
        self.jaccard = jaccard
        self.f1_score = f1_score
        self.aed_score = aed_score
        self.confusion_matrix = confusion_matrix
        self.confidence_intervals: Tuple[float, float] = confidence_intervals

    def to_string(self):
        metrics = "accuracy (A): " + str(self.accuracy)
        if self.balanced_accuracy:
            metrics += "\nbalanced_accuracy (BA): " + str(self.balanced_accuracy)
        if self.balanced_accuracy_adjusted:
            metrics += "\nbalanced_accuracy_adjusted (BA): " + str(self.balanced_accuracy_adjusted)
        if self.precision:
            metrics += "\nprecision (P): " + str(self.precision)
        if self.average_precision:
            metrics += "\naverage_precision (AP): " + str(self.average_precision)
        if self.recall:
            metrics += "\nrecall (R): " + str(self.recall)
        if self.jaccard:
            metrics += "\njaccard (IoU): " + str(self.jaccard)
        if self.f1_score:
            metrics += "\nf1_score (F): " + str(self.f1_score)
        if self.aed_score:
            metrics += "\naed_score (AED): " + str(self.aed_score)
        if self.confusion_matrix is not None:
            metrics += "\nconfusion_matrix (C):\n" + str(self.confusion_matrix)
        return metrics

    def save_to_file(self, path):
        from utils.file_management import save_data_with_ultimate_dir_creation
        save_data_with_ultimate_dir_creation(path, [self.to_string()])


def aed_score(y_true, y_pred):
    import numpy as np
    from utils.mapping import letters_to_ints
    if type(y_true[0]) == str or type(y_true[0]) == np.str_:
        y_true = letters_to_ints(y_true)
    if type(y_pred[0]) == str or type(y_pred[0]) == np.str_:
        y_pred = letters_to_ints(y_pred)
    return np.mean([abs(t - p) / 2 for t, p in zip(y_true, y_pred)])


def confidence_intervals(accuracies: List[float]) -> Tuple[float, float]:
    import numpy as np, scipy.stats as st
    if st.sem(accuracies) == 0:
        return accuracies[0], accuracies[0]
    return st.t.interval(0.95, len(accuracies) - 1, loc=np.mean(accuracies), scale=st.sem(accuracies))


def calculate_metrics(y_true: List, y_pred: List) -> Metrics:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, jaccard_score, f1_score, \
        balanced_accuracy_score, confusion_matrix
    from utils.mapping import proba_to_letters, letters_to_ints, floats_to_ints
    import numpy as np
    if type(y_true[0]) in [List, np.ndarray, list]:
        y_true = proba_to_letters(y_true)
    if type(y_pred[0]) in [List, np.ndarray, list]:
        y_pred = proba_to_letters(y_pred)

    if type(y_pred[0]) in [np.str_, str]:
        y_pred = letters_to_ints(y_pred)
    if type(y_true[0]) in [np.str_, str]:
        y_true = letters_to_ints(y_true)

    if type(y_pred[0]) in [np.float64, float]:
        y_pred = floats_to_ints(y_pred)

    # print(type(y_true), type(y_pred), type(y_true[0]), type(y_pred[0]), y_true[0], y_pred[0])

    m = Metrics(accuracy=accuracy_score(y_true, y_pred),
                balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
                balanced_accuracy_adjusted=balanced_accuracy_score(y_true, y_pred, adjusted=True),
                precision=precision_score(y_true, y_pred, average='weighted'),
                recall=recall_score(y_true, y_pred, average='weighted'),
                jaccard=jaccard_score(y_true, y_pred, average='weighted'),
                f1_score=f1_score(y_true, y_pred, average='weighted'),
                aed_score=aed_score(y_true, y_pred),
                confusion_matrix=confusion_matrix(y_true, y_pred))

    return m


def get_y_pred_y_true(y_pred_path: str, y_true_path: str) -> Tuple[List[int], List[int]]:
    def process_lines(file) -> List[int]:
        lines = file.readlines()
        els = map(lambda x: x.strip(), lines)
        els = filter(lambda x: x != '', els)
        els = list(map(int, els))
        return els

    with open(y_pred_path, 'r', encoding="utf-8") \
            as y_pred_file, open(y_true_path, 'r', encoding="utf-8") as y_true_file:
        y_pred = process_lines(y_pred_file)
        y_true = process_lines(y_true_file)
        return y_pred, y_true


def get_metrics_from_files(y_pred_path: str, y_true_path: str) -> Metrics:
    y_pred, y_true = get_y_pred_y_true(y_pred_path, y_true_path)
    metrics = calculate_metrics(y_true, y_pred)

    return metrics


def save_metrics(y_pred: List, y_true: List, metrics_path: str) -> None:
    metrics = calculate_metrics(y_true, y_pred)
    print(metrics.to_string())
    metrics.save_to_file(metrics_path)


def save_metrics_from_files(y_pred_path: str, y_true_path: str, metrics_path: str) -> None:
    y_pred, y_true = get_metrics_from_files(y_pred_path, y_true_path)
    save_metrics(y_pred, y_true, metrics_path)


class Result:
    def __init__(self, metrics: Metrics, params: Params):
        self.classifier: str = params.classifier
        self.metrics: Metrics = metrics
        self.features: List[str] = params.get_features_names_list()
        self.dirname: str = params.dirname
        self.stacked: bool = params.stacked
        self.balanced: bool = params.params['balanced'] if 'balanced' in params.params.keys() else False
        self.to_binary: bool = params.to_binary
        self.use_only_ab: bool = params.use_only_ab

    def get_features_str(self):
        return '__'.join(sorted(self.features))

    def save_result(self):
        from ebm.filenames import get_metrics_path
        features_str = self.get_features_str()
        prefix = ''
        if self.balanced:
            prefix += 'balanced_'
        if self.to_binary:
            prefix += 'binary_'

        suffix = ''
        if self.use_only_ab:
            suffix = '_using_only_ab'

        method = self.dirname + suffix
        trainer = prefix + self.classifier

        path = get_metrics_path(features_str, trainer, method)

        print(features_str + ' ' + self.classifier + ' ' + self.dirname + ':\n' + self.metrics.to_string() + '\n')
        self.metrics.save_to_file(path)

    def to_string(self):
        lines = [self.metrics.to_string()]

        prefix = ''
        if self.balanced:
            prefix += 'balanced_'
        if self.to_binary:
            prefix += 'binary_'

        suffix = ''
        if self.use_only_ab:
            suffix = '_using_only_ab'

        method = self.dirname + suffix
        trainer = prefix + self.classifier

        lines.append(method)
        lines.append(trainer)
        lines += self.features

        return '\n'.join(lines)


def read_test_result_from_metrics_file(path: str) -> Result:
    els = path.split('/')
    filename = els[-1]
    classifier = els[-2]
    dirname = els[-3]
    stacked = '_on_' in dirname
    balanced = 'balanced_' in classifier
    to_binary = 'binary_' in classifier
    use_only_ab = '_using_only_ab' in dirname
    features_str = filename[:-len('_metrics.txt')]
    features = features_str.split('__')
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        accuracy = float(lines[0].split(': ')[1])
        f1_score = None
        balanced_accuracy_adjusted = None
        for line in lines:
            if 'f1_score' in line:
                f1_score = float(line.split(': ')[1])
            if 'balanced_accuracy_adjusted' in line:
                balanced_accuracy_adjusted = float(line.split(': ')[1])

        metrics = Metrics(accuracy, balanced_accuracy_adjusted=balanced_accuracy_adjusted, f1_score=f1_score)
        # TODO read f1 scores and more here as well
        # TODO pca is hardset to False

        return Result(metrics=metrics,
                      params=Params(features=features, classifier=classifier, dirname=dirname, stacked=stacked,
                                    params={'balanced': balanced}, to_binary=to_binary, use_only_ab=use_only_ab,
                                    pca=False))


def filter_only_metrics_filenames(filepaths: List[str]) -> List[str]:
    return list(filter(lambda x: 'metrics.txt' in x, filepaths))


def filter_filepaths_that_have_highest_accuracies(filepaths: List[str], n=10) -> Tuple[List[str], List[float]]:
    filepaths = filter_only_metrics_filenames(filepaths)

    results = {}
    for path in filepaths:
        result = read_test_result_from_metrics_file(path=path)
        results[path] = result

    l = sorted([val.metrics.accuracy for val in results.values()])[max(0, len(results) - n):]
    min_acc = l[0]

    def f(path):
        result = read_test_result_from_metrics_file(path=path)
        return result.metrics.accuracy >= min_acc

    filepaths = list(filter(f, filepaths))
    accuracies = [results[f].metrics.accuracy for f in filepaths]

    return filepaths, accuracies


def filter_filepaths_with_highest_accuracies(feature_name, directories: List[str]) -> Result:
    from utils.file_management import get_all_filepaths_in_dirs
    filepaths = get_all_filepaths_in_dirs(directories)
    filepaths = filter_only_metrics_filenames(filepaths)

    results = {}
    for path in filepaths:
        if feature_name in path:
            result = read_test_result_from_metrics_file(path=path)
            results[path] = result

    return sorted([val for val in results.values()], key=lambda result: result.metrics.accuracy)[-1]
