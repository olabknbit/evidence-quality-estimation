from ensemble_methods.ensemble_sklearn import run
from utils.feature_names import *

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
    # # AdaBoostClassifier_name,
    # # BernouliNB_name,
    # # DecisionTreeClassifier_name,
    # ExtraTreesClassifier_name,
    # # ExtraTreesRegressor_name,
    # # GaussianNB_name,
    # GradientBoostingClassifier_name,
    # # KNeighborsClassifier_name,
    # # LinearSVC_name,
    # MLPClassifier_name,
    # MLPRegressor_name,
    # StackingClassifier_name,
    # # SVC_with_linear_kernel_name,
    # # SVC_name,
    # # VotingClassifier_name,
    RandomForestClassifier_name,
    # RandomForestClassifier500_name,
    # # RandomForestRegressor_name,
    # # SGDClassifier_name,
]


def run_single():
    from ensemble_methods.feature_sets_for_ensemble import get_all_highest_perf_single_feature_subsets_for_ensemble
    all_subsets = get_all_highest_perf_single_feature_subsets_for_ensemble()

    n_features = len(all_subsets)
    n_classifiers = len(classifier_names)

    print(n_features)
    max_a = 0.
    for i_bal, balanced in enumerate([False]):
        for i_ab, use_only_ab in enumerate([True, False]):
            for i_binary, binary in enumerate([False, True]):
                for ic, classifier in enumerate(classifier_names):
                    if balanced and classifier in classifiers_not_supporting_balancing:
                        continue
                    for ifnl, features in enumerate(all_subsets):
                        print('Balanced', balanced)
                        print('Use only ab', use_only_ab)
                        print('Binary', binary)
                        print('Classifier: ', ic, ' out of ', n_classifiers, classifier)
                        print('Features ', ifnl, ' out of ', n_features)
                        print('Progress ', (
                                ((((i_bal * 2 + i_ab) * 2 + i_binary) * 2 + ic) * n_features) + ifnl)
                              / (n_classifiers * n_features * 2 * 2 * 2 * 2) * 100., '%')
                        a = run(features, classifier, to_binary=binary,
                                use_only_ab=use_only_ab, balanced=balanced)
                        max_a = max(max_a, a)
                        print("max accuracy: %0.4f current accuracy %0.4f\n" % (max_a, a))


def run_multi():
    from ensemble_methods.feature_sets_for_ensemble import get_all_highest_perf_multiple_feature_subsets_for_ensemble
    all_subsets = get_all_highest_perf_multiple_feature_subsets_for_ensemble()

    n_features = len(all_subsets)
    n_classifiers = len(classifier_names)

    print(n_features)
    max_a = 0.
    for i_bal, balanced in enumerate([False]):
        for i_ab, use_only_ab in enumerate([True, False]):
            for i_binary, binary in enumerate([False, True]):
                for ic, classifier in enumerate(classifier_names):
                    if balanced and classifier in classifiers_not_supporting_balancing:
                        continue
                    for ifnl, feature_names_list in enumerate(all_subsets):
                        print('Balanced', balanced)
                        print('Use only ab', use_only_ab)
                        print('Binary', binary)
                        print('Classifier: ', ic, ' out of ', n_classifiers, classifier)
                        print('Features ', ifnl, ' out of ', n_features)
                        print('Progress ', (
                                ((((i_bal * 2 + i_ab) * 2 + i_binary) * 2 + ic) * n_features) + ifnl)
                              / (n_classifiers * n_features * 2 * 2 * 2 * 2) * 100., '%')
                        mea = run(feature_names_list, classifier, to_binary=binary, use_only_ab=use_only_ab,
                                  balanced=balanced)

                        with open("out/metrics/stacked_mixed_sklearn_metrics/sklearn/" + classifier + ".txt", 'a') as f:
                            f.write(str(mea))
                            f.write('\n---------------------\n')
                        max_a = max(max_a, mea)
                        print("max accuracy: %0.4f current accuracy %0.4f\n" % (max_a, mea))


if __name__ == "__main__":
    pass
