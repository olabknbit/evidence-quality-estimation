from base_methods.base_sklearn import generate_probabilities
from ebm.dataset import get_train_dev_data, get_test_data
from params import MultiBaseParams
from params import SingleBaseParams
from utils.feature_names import *

classifier_names = [
    AdaBoostClassifier_name,
    BernouliNB_name,
    DecisionTreeClassifier_name,
    ExtraTreesClassifier_name,
    ExtraTreesRegressor_name,
    GaussianNB_name,
    GradientBoostingClassifier_name,
    KNeighborsClassifier_name,
    LinearSVC_name,
    MLPClassifier_name,
    MLPRegressor_name,
    SGDClassifier_name,
    StackingClassifier_name,
    SVC_with_linear_kernel_name,
    SVC_name,
    VotingClassifier_name,
    RandomForestClassifier_name,
    RandomForestClassifier500_name,
    RandomForestRegressor_name,
]


def single_features(test_only=False):
    train = get_train_dev_data()
    test = get_test_data()

    for classifier in classifier_names:
        for feature in [publication_type_key]:
            print(feature)
            params = SingleBaseParams(feature_name=feature, classifier=classifier, dirname="new", params={}, pca=False)
            generate_probabilities(train=train, test=test, params=params, test_only=test_only)
            params.serialize()


def multiple_features(test_only=False):
    train = get_train_dev_data()
    test = get_test_data()

    all_subsets = get_all_feature_subsets()
    for classifier in classifier_names:
        for subset in all_subsets:
            print(subset)
            params = MultiBaseParams(features=subset, classifier=classifier, dirname="new", params={}, pca=False)
            generate_probabilities(train, test, params, test_only=test_only)
            params.serialize()


if __name__ == "__main__":
    pass
