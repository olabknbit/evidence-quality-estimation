from typing import Dict, Any

from utils.feature_names import *


def run_LinearSVC(params: Dict[str, Any]):
    from sklearn import svm

    clf = svm.LinearSVC(**params, max_iter=10000)
    return clf


def run_SVC(params: Dict[str, Any]):
    from sklearn import svm
    params['probability'] = True
    params['gamma'] = 'scale'

    clf = svm.SVC(**params)
    return clf


def run_SVC_with_linear_kernel(params: Dict[str, Any]):
    params['kernel'] = 'linear'
    return run_SVC(params=params)


def run_GradientBoostingClassifier(params: Dict[str, Any]):
    from sklearn.ensemble import GradientBoostingClassifier

    clf = GradientBoostingClassifier(**params)
    return clf


def run_SGDClassifier(params: Dict[str, Any]) \
        :
    from sklearn.linear_model import SGDClassifier

    clf = SGDClassifier(**params)
    return clf


def run_KNeighborsClassifier(params: Dict[str, Any]) \
        :
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(**params)
    return clf


def run_MLPClassifier(params: Dict[str, Any]) \
        :
    from sklearn.neural_network import MLPClassifier
    params["max_iter"] = 2000
    clf = MLPClassifier(**params)
    return clf


def run_MLPRegressor(params: Dict[str, Any]) \
        :
    from sklearn.neural_network import MLPRegressor
    params["max_iter"] = 2000
    clf = MLPRegressor(**params)
    return clf


def run_bernouliNB(params: Dict[str, Any]):
    from sklearn.naive_bayes import BernoulliNB

    clf = BernoulliNB(**params)
    return clf


def run_gaussianNB(params: Dict[str, Any]):
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB(**params)
    return clf


def run_RandomForestClassifier(params: Dict[str, Any]):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(**params)
    return clf


def run_RandomForestClassifier500(
        params: Dict[str, Any]):
    params["n_estimators"] = 500
    return run_RandomForestClassifier(params)


def run_RandomForestRegressor(params: Dict[str, Any]):
    from sklearn.ensemble import RandomForestRegressor
    clf = RandomForestRegressor(**params)
    return clf


def run_ExtraTreesRegressor(params: Dict[str, Any]):
    from sklearn.ensemble import ExtraTreesRegressor

    clf = ExtraTreesRegressor(**params)
    return clf


def run_VotingClassifier(params: Dict[str, Any]):
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    clf1 = LogisticRegression(multi_class='multinomial')
    clf2 = RandomForestClassifier(n_estimators=50)
    clf3 = SVC(kernel="linear")

    clf = VotingClassifier(**params, estimators=[('lr', clf1), ('rf', clf2), ("svc", clf3)])
    return clf


def run_StackingClassifier(params: Dict[str, Any]):
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    clf1 = LogisticRegression(multi_class='multinomial')
    clf2 = RandomForestClassifier(n_estimators=50)
    clf3 = SVC()

    clf = StackingClassifier(**params, estimators=[('lr', clf1), ('rf', clf2), ("svc", clf3)])
    return clf


def run_ExtraTreesClassifier(params: Dict[str, Any]):
    from sklearn.ensemble import ExtraTreesClassifier

    clf = ExtraTreesClassifier(**params)
    return clf


def run_DecisionTreeClassifier(params: Dict[str, Any]):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(**params)
    return clf


def run_AdaBoostClassifier(params: Dict[str, Any]):
    from sklearn.ensemble import AdaBoostClassifier

    clf = AdaBoostClassifier(**params)
    return clf


def get(classifier_name: str, params: Dict[str, Any]):
    classifier_names = {
        AdaBoostClassifier_name: run_AdaBoostClassifier,
        BernouliNB_name: run_bernouliNB,
        DecisionTreeClassifier_name: run_DecisionTreeClassifier,
        ExtraTreesClassifier_name: run_ExtraTreesClassifier,
        ExtraTreesRegressor_name: run_ExtraTreesRegressor,
        GaussianNB_name: run_gaussianNB,
        GradientBoostingClassifier_name: run_GradientBoostingClassifier,
        KNeighborsClassifier_name: run_KNeighborsClassifier,
        LinearSVC_name: run_LinearSVC,
        MLPClassifier_name: run_MLPClassifier,
        MLPRegressor_name: run_MLPRegressor,
        RandomForestClassifier_name: run_RandomForestClassifier,
        RandomForestClassifier500_name: run_RandomForestClassifier500,
        RandomForestRegressor_name: run_RandomForestRegressor,
        SGDClassifier_name: run_SGDClassifier,
        StackingClassifier_name: run_StackingClassifier,
        SVC_name: run_SVC,
        SVC_with_linear_kernel_name: run_SVC_with_linear_kernel,
        VotingClassifier_name: run_VotingClassifier,
    }
    if classifier_name not in classifier_names.keys():
        print("classifier not recognized", classifier_name)
        exit(1)
    f = classifier_names[classifier_name]
    clf = f(params=params)
    return clf
