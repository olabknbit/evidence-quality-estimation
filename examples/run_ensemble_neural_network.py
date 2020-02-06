from ensemble_methods.ensemble_keras import run

optimizers = [
    'adadelta',
    'adagrad',
    'nadam',
    'adam',
    'adamax',
    'rmsprop',
]


def run_single():
    from ensemble_methods.feature_sets_for_ensemble import get_all_highest_perf_single_feature_subsets_for_ensemble
    all_subsets = get_all_highest_perf_single_feature_subsets_for_ensemble()
    max_a = 0.

    for use_only_ab in [True, False]:
        for ic, optimizer in enumerate(optimizers):
            # for alpha in range(2, 10):
            # alpha = 2
            for ifnl, feature_names_list in enumerate(all_subsets):
                print(feature_names_list)
                a = run(feature_names_list, optimizer, use_only_ab=use_only_ab)
                max_a = max(max_a, a)
                # print('alpha:', alpha)
                print('Use only ab', use_only_ab)
                print('Classifier: ', ic, ' out of ', len(optimizers), optimizer)
                print('Features ', ifnl, ' out of ', len(all_subsets))
                print('Progress ', ((ic * len(all_subsets)) + ifnl) / (len(optimizers) * len(all_subsets)) * 100., '%')
                print("max accuracy: %0.4f current accuracy %0.4f\n" % (max_a, a))


def run_multiple():
    from ensemble_methods.feature_sets_for_ensemble import get_all_highest_perf_multiple_feature_subsets_for_ensemble
    all_subsets = get_all_highest_perf_multiple_feature_subsets_for_ensemble()

    n_features = len(all_subsets)
    n_classifiers = len(optimizers)

    print(n_features)
    max_a = 0.

    for i_ab, use_only_ab in enumerate([True, False]):
        for ic, classifier in enumerate(optimizers):
            for ifnl, feature_names_list in enumerate(all_subsets):
                print('Use only ab', use_only_ab)

                print('Classifier: ', ic, ' out of ', n_classifiers, classifier)
                print('Features ', ifnl, ' out of ', n_features)
                print('Progress ', (
                        ((((i_ab)) * 2 + ic) * n_features) + ifnl)
                      / (n_classifiers * n_features * 2 * 2) * 100., '%')
                a = run(feature_names_list, classifier,
                        use_only_ab=use_only_ab)
                with open("out/metrics/stacked_mixed_sklearn_metrics/keras/" + classifier + ".txt", 'a') as f:
                    f.write(str(a))
                    f.write('\n---------------------\n')
                max_a = max(max_a, a)
                print("max accuracy: %0.4f current accuracy %0.4f\n" % (max_a, a))


if __name__ == "__main__":
    pass
