from ensemble_methods.feature_sets_for_ensemble import get_all_highest_perf_multiple_feature_subsets_for_ensemble, \
    get_all_highest_perf_single_feature_subsets_for_ensemble


def auctioning_single():
    from ensemble_methods.ensemble_auctioning import run

    all_subsets = get_all_highest_perf_single_feature_subsets_for_ensemble()

    max_a = 0.
    for ifnl, feature_list in enumerate(all_subsets):
        a = run(feature_list, verbose=False, with_save=False, name="new")
        max_a = max(max_a, a)
        print('Features ', ifnl, ' out of ', len(all_subsets))
        print('Progress ', ifnl / len(all_subsets) * 100., '%')
        print("max accuracy: %0.4f current accuracy %0.4f\n" % (max_a, a))


def consensus_single(with_save):
    from ensemble_methods.ensemble_consensus import run
    all_subsets = get_all_highest_perf_single_feature_subsets_for_ensemble()

    max_a = 0.
    lines = []
    for ifnl, features in enumerate(all_subsets):
        a, line = run(features, name="new", verbose=False, with_save=with_save)
        max_a = max(max_a, a)
        print('Features ', ifnl, ' out of ', len(all_subsets))
        print('Progress ', ifnl / len(all_subsets) * 100., '%')
        print("max accuracy: %0.4f current accuracy %0.4f\n" % (max_a, a))
        lines.append(line)

    lines = list(reversed(sorted(lines, key=lambda x: x[1])))
    with open("out/consensus_single.txt", "a") as f:
        f.writelines(["%s & %.2f & %.2f \\\\\n\\hline\n" % l for l in lines])


def game_theory_single(with_save=False):
    from ensemble_methods.ensemble_game_theory import run
    all_subsets = get_all_highest_perf_single_feature_subsets_for_ensemble()

    max_a = 0.
    lines = []
    for ifnl, features in enumerate(all_subsets):
        a, line = run(features, verbose=False, with_save=with_save)
        max_a = max(max_a, a)
        print('Features ', ifnl, ' out of ', len(all_subsets))
        print('Progress ', ifnl / len(all_subsets) * 100., '%')
        print("max accuracy: %0.4f current accuracy %0.4f\n" % (max_a, a))
        lines.append(line)

    lines = list(reversed(sorted(lines, key=lambda x: x[1])))
    with open("out/game_theory_single.txt", "a") as f:
        f.writelines(["%s & %.2f & %.2f \\\\\n\\hline\n" % l for l in lines])


def consensus_multiple(with_save=False, with_scaling=False):
    from ensemble_methods.ensemble_consensus import run
    all_subsets = get_all_highest_perf_multiple_feature_subsets_for_ensemble()

    max_a = 0.
    lines = []
    for ifnl, features in enumerate(all_subsets):
        a, line = run(features, name="new", verbose=False, with_save=with_save,
                      with_scaling=with_scaling)
        max_a = max(max_a, a)
        print('Features ', ifnl, ' out of ', len(all_subsets))
        print('Progress ', ifnl / len(all_subsets) * 100., '%')
        print("max accuracy: %0.4f current accuracy %0.4f\n" % (max_a, a))
        lines.append(line)
        lines = list(reversed(sorted(lines, key=lambda x: x[1])))
        filename = "out/consensus_multiple.txt"
        if with_scaling:
            filename = "out/consensus_multiple_with_scaling.txt"
        with open(filename, "w") as f:
            f.writelines(["%s & %.2f & %.2f \\\\\n\\hline\n" % l for l in lines])


def game_theory_multiple(with_save=False, with_scaling=False):
    from ensemble_methods.ensemble_game_theory import run

    all_subsets = get_all_highest_perf_multiple_feature_subsets_for_ensemble()

    max_a = 0.
    lines = []
    for ifnl, features in enumerate(all_subsets):
        a, line = run(features, verbose=False, with_save=with_save,
                      with_scaling=with_scaling)
        max_a = max(max_a, a)
        print('Features ', ifnl, ' out of ', len(all_subsets))
        print('Progress ', ifnl / len(all_subsets) * 100., '%')
        print("max accuracy: %0.4f current accuracy %0.4f\n" % (max_a, a))
        lines.append(line)
        lines = list(reversed(sorted(lines, key=lambda x: x[1])))
        filename = "out/game_theory_multiple.txt"
        if with_scaling:
            filename = "out/game_theory_multiple_with_scaling.txt"
        with open(filename, "w") as f:
            f.writelines(["%s & %.2f & %.2f \\\\\n\\hline\n" % l for l in lines])


def analyse_scores_multiple(with_scaling):
    lines = None
    filename = "out/game_theory_multiple.txt"
    if with_scaling:
        filename = "out/game_theory_multiple_with_scaling.txt"
    with open(filename, "r") as f:
        lines = f.read()[:-2]
    lines = lines.split('\\hline')
    lines = list(set(lines))

    def strip_line(line):
        elems = line.split(' & ')
        return [e.strip() for e in elems]

    lines = [strip_line(l) for l in lines]
    print(lines[:10])
    lines = list(reversed(sorted(lines, key=lambda x: x[1])))
    tr_lines = [' & '.join(line) for line in lines]
    for line in tr_lines[:10]:
        print(line)
    print()
    lines = list(reversed(sorted(lines, key=lambda x: x[2])))
    te_lines = [' & '.join(line) for line in lines]
    for line in te_lines[:10]:
        print(line)


if __name__ == "__main__":
    pass
