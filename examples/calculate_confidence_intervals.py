import numpy as np

from base_methods.base_sklearn import generate_probabilities
from ebm.dataset import get_train_dev_data, get_test_data
from params import MultiBaseParams, SingleBaseParams
from utils.feature_names import *
from utils.metrics import confidence_intervals
from utils.stacking import get_feature_name_sets_with_highest_probable_accuracy

classifiers_not_supporting_balancing = [
    AdaBoostClassifier_name,
    BernouliNB_name,
    ExtraTreesRegressor_name,
    GaussianNB_name,
    GradientBoostingClassifier_name,
    KNeighborsClassifier_name,
    MLPClassifier_name,
    MLPRegressor_name,
    StackingClassifier_name,
    VotingClassifier_name,
    RandomForestRegressor_name,
]

classifier_names = [
    # AdaBoostClassifier_name,
    # BernouliNB_name,
    # DecisionTreeClassifier_name,
    # ExtraTreesClassifier_name,
    # ExtraTreesRegressor_name,
    # GaussianNB_name,
    # GradientBoostingClassifier_name,
    # KNeighborsClassifier_name,
    # LinearSVC_name,
    # MLPClassifier_name,
    # MLPRegressor_name,
    # SGDClassifier_name,
    # StackingClassifier_name,
    # SVC_with_linear_kernel_name,
    SVC_name,
    # VotingClassifier_name,
    RandomForestClassifier_name,
    # RandomForestClassifier500_name,
    # RandomForestRegressor_name,
]


def single_features():
    train = get_train_dev_data()
    test = get_test_data()

    for i_bal, balanced in enumerate([False, True]):
        for classifier in classifier_names:
            if balanced and classifier in classifiers_not_supporting_balancing:
                continue
            for feature_name in all_feature_names:
                print(feature_name)
                accuracies = []
                for i in range(15):
                    params = SingleBaseParams(feature_name, classifier, dirname=sklearn_master_name,
                                              params={'balanced': balanced}, pca=False)
                    a = generate_probabilities(train, test, params, test_only=True)
                    print('\t', i, feature_name, classifier, a)
                    accuracies.append(a * 100)

                l, r = confidence_intervals(accuracies)
                m = float(np.mean([l, r]))
                balanced_str = "balanced" if balanced else "not_balanced"
                results = "%s & %s & %s & %.3f & %.1f-%.1f" % (balanced_str, feature_name, classifier, m, l, r)
                print(results)
                with open("out/conf_int/base/" + feature_name + ".txt", 'a') as f:
                    f.write("%s\n" % results)


def multiple_features(dirname):
    train = get_train_dev_data()
    test = get_test_data()

    all_subsets = get_feature_name_sets_with_highest_probable_accuracy(dirname)

    n_features = len(all_subsets)
    n_classifiers = len(classifier_names)

    print(n_features)
    max_a = 0.
    for i_bal, balanced in enumerate([False]):
        for ic, classifier in enumerate(classifier_names):
            if balanced and classifier in classifiers_not_supporting_balancing:
                continue
            for ifnl, feature_names_list in enumerate(all_subsets):
                print('Balanced', balanced)

                print('Classifier: ', ic, ' out of ', n_classifiers, classifier)
                print('Features ', ifnl, ' out of ', n_features)
                print('Progress ', (
                        ((((i_bal)) * 2 + ic) * n_features) + ifnl)
                      / (n_classifiers * n_features * 2 * 2) * 100., '%')
                accs = []
                for i in range(10):
                    params = MultiBaseParams(features=feature_names_list, classifier=classifier,
                                             dirname=dirname, pca=False, params={'balanced': balanced})
                    a = generate_probabilities(train, test, params, test_only=True)
                    accs.append(a * 100)

                l, r = confidence_intervals(accs)
                m = float(np.mean([l, r]))
                balanced_str = "balanced" if balanced else "not_balanced"
                results = "%s & %s & %s & %.3f & %.1f-%.1f" % (
                    balanced_str, ' '.join(feature_names_list), classifier, m, l, r)
                print(results)
                with open("out/conf_int/" + dirname + "/all.txt", 'a') as f:
                    f.write("%s\n" % results)

                max_a = max(max_a, m)
                print("max accuracy: %0.4f current accuracy %0.4f\n" % (max_a, m))


if __name__ == "__main__":
    pass
