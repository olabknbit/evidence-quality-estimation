from typing import List

import numpy as np

from params import BaseParams, StackedParams
from utils.mapping import map_y_array_to_int
from utils.metrics import calculate_metrics


class Agent:
    def __init__(self, x: List[List[float]], y: List[str], feature: BaseParams, verbose, with_scaling):
        self.x = x
        self._i = 0

        self.y = map_y_array_to_int(y)
        self.feature = feature
        self.feature_name = feature.get_features_names_str()

        self.verbose = verbose
        self.accuracy = calculate_metrics(self.y, x).accuracy
        print(self.feature_name, self.accuracy)

        self.with_scaling = with_scaling

    def get_scores(self):
        l = self.x[self._i]
        if self.with_scaling:
            l = [x * self.accuracy for x in l]
        return l

    def rank(self, sor: int):
        grades = list(reversed(sorted([(i, s) for i, s in enumerate(self.get_scores())], key=lambda x: x[1])))
        grades = [(i, s if s != 0 else 0.01) for i, s in grades]
        grades = {grade: (ith + 1 / score) for ith, (grade, score) in enumerate(grades)}

        return grades[sor]


def get_x_y(feature: BaseParams, verbose, train_vs_test, with_scaling=False):
    from features.base_features import read_one_feature
    _, x_train, y_train = read_one_feature(feature, train_vs_test=train_vs_test, only_ab=False, to_binary=False)
    return Agent(x_train, y_train, feature, verbose=verbose, with_scaling=with_scaling)


class Game:
    def __init__(self, agents: List[Agent], verbose):
        self.agents = agents
        self.verbose = verbose
        self._i = 0

    def upp_i(self):
        self._i += 1
        for a in self.agents:
            a._i = self._i

    def small_round(self):
        As = [a.rank(0) for a in self.agents]
        Bs = [a.rank(1) for a in self.agents]
        Cs = [a.rank(2) for a in self.agents]

        averages = [np.average(As), np.average(Bs), np.average(Cs)]
        averages = list(sorted([(i, av) for i, av in enumerate(averages)], key=lambda x: x[1]))
        return averages[0][0]

    def play(self):
        predictions = []
        for i in range(len(self.agents[0].x)):
            if self.verbose:
                print("\n----------------------")
                print("round:", i)
                print("----------------------\n")
            winner = self.small_round()
            if self.verbose:
                print(winner, "TRUE:", self.agents[0].y[i])
            predictions.append(winner)
            self.upp_i()
        y = self.agents[0].y
        from utils.metrics import calculate_metrics
        return calculate_metrics(y, predictions)


def create_game_from_feature_names(features, verbose=False, mode="test", with_scaling=False):
    agents = [get_x_y(feature, verbose, mode, with_scaling) for i, feature in enumerate(features)]
    return Game(agents, verbose)


def run(features: List[BaseParams], name, verbose=False, with_save=False, with_scaling=False):
    game = create_game_from_feature_names(features, verbose, mode="train", with_scaling=with_scaling)
    m_train = game.play()

    game = create_game_from_feature_names(features, verbose, mode="test", with_scaling=with_scaling)
    m_test = game.play()

    new_fnames = [fn.get_features_names_str() for fn in features]
    feature_names_str = ', '.join(new_fnames)

    from utils.metrics import Result
    params = StackedParams(features=features, classifier="consensus", dirname="new", pca=False, use_only_ab=False,
                           to_binary=False, params={'with_scaling': with_scaling})
    r = Result(metrics=m_test, params=params)
    print(m_test.to_string())
    if with_save:
        r.save_result()
    return m_test.accuracy, (feature_names_str, m_train.accuracy * 100, m_test.accuracy * 100)
