import os.path

import numpy as np

from ebm.dataset import Data
from ebm.filenames import *
from params import MultiBaseParams, BaseParams, SingleBaseParams
from utils.feature_names import *
from utils.file_management import save_data_with_ultimate_dir_creation


def serialize_features_f(evidence_ids: List[str], X: List[List[float]], y: List[str], params: BaseParams,
                         test_vs_train: str) -> None:
    for (evidence_id, features, sor) in zip(evidence_ids, X, y):
        features = [str(f) for f in features]
        features = [' '.join(features)]
        filename = evidence_id + '.txt'
        path = os.path.join(
            get_features_dir(params.get_features_names_str(), params.dirname, test_vs_train, params.classifier),
            sor,
            filename)
        save_data_with_ultimate_dir_creation(path, features)


def get_Xy(data: Data, params: BaseParams):
    from features.get_from_reference import get_f, get_fs
    if type(params) == MultiBaseParams:
        params: MultiBaseParams
        fs = get_fs(params.features)
        X, y = data.get_pandas_xy(fs)
    elif type(params) == SingleBaseParams:
        params: SingleBaseParams
        f = get_f(params.get_feature_name())
        X, y = data.get_x_y(f)
    return X, y


def get_pipeline(params: BaseParams):
    if type(params) == MultiBaseParams:
        params: MultiBaseParams
        from base_methods.base_sklearn_multiple_features import get_features_pipeline
        pipeline = get_features_pipeline(params)
    elif type(params) == SingleBaseParams:
        params: SingleBaseParams
        from base_methods.base_sklearn_single_features import get_feature_pipeline
        pipeline = get_feature_pipeline(params)
    return pipeline


def train_SVC_pipeline(X_train, X_test, y_train, y_test, pipeline):
    pipeline.fit(X_train, y_train)
    print("score = %3.4f" % (pipeline.score(X_test, y_test)))
    return pipeline.predict_proba(X_test)


def h(train_data, test_data, params: BaseParams, trues_, probs_, test_only, test_vs_train):
    X_train, y_train = get_Xy(train_data, params)
    X_test, y_test = get_Xy(test_data, params)
    pipeline = get_pipeline(params)

    probs = train_SVC_pipeline(X_train, X_test, y_train, y_test, pipeline)
    trues_ = np.concatenate((trues_, y_test), axis=0)
    probs_ = np.concatenate((probs_, probs), axis=0)

    if not test_only:
        serialize_features_f(evidence_ids=test_data.get_evidence_ids(), X=probs, y=y_test, params=params,
                             test_vs_train=test_vs_train)
    return trues_, probs_


def generate_probabilities(train: Data, test: Data, params: BaseParams, test_only=False):
    probs_ = np.array([]).reshape(0, 3)
    trues_ = np.array([])
    if not test_only:
        n = 10
        k = int(len(train.snippets) / n) + 1

        for i in range(k):
            train_data, test_data = train.split_batches(n, i)
            print("%d/%d (%3.2f%%)" % (i, k, i / k * 100))

            trues_, probs_ = h(train_data, test_data, params, trues_, probs_, test_only, "train")

    trues_, probs_ = h(train, test, params, trues_, probs_, test_only, "test")

    from utils.metrics import Result, calculate_metrics
    from utils.mapping import proba_to_letters
    metrics = calculate_metrics(trues_, proba_to_letters(probs_))

    r = Result(metrics=metrics, params=params)
    if not test_only:
        r.save_result()

    return r.metrics.accuracy
