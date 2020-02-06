from typing import List, Tuple, Optional

from ebm.filenames import game_theory_name
from params import BaseParams, StackedParams
from utils.metrics import Result

keep = 'keep'
change = 'change'
any_ = "any"


class Agent:
    def __init__(self, x: List[List[float]], y: List[str], feature: BaseParams, verbose, with_scaling):
        self.x = x
        from utils.mapping import map_y_array_to_int
        from utils.metrics import calculate_metrics
        self.y = map_y_array_to_int(y)
        self.feature = feature
        self.feature_name = feature.get_features_names_str()
        self.updated_ranking = dict()
        self.updated_ranking_changed: int = 0
        self.verbose = verbose
        self.accuracy = calculate_metrics(self.y, x).accuracy
        self.with_scaling = with_scaling

    def reset(self):
        self.updated_ranking = dict()
        self.updated_ranking_changed = 0

    def get_rank(self, sor_grade, i):
        return self.get_scores(i)[sor_grade]

    def find_highest_prob(self, _i) -> float:
        highest_prob = 0.
        a, b, c = self.get_scores(_i)
        curr = max(a, b, c)
        if curr > highest_prob:
            highest_prob = curr
        return highest_prob

    def find_highest_sor_grade(self, _i) -> int:
        grades = list(reversed(sorted([(i, s) for i, s in enumerate(self.get_scores(_i))], key=lambda x: x[1])))
        return grades[0][0]

    def update_ranking(self, sor_grade, val, changed, i):
        finish = "SECOND TIME"
        if changed == change:
            self.updated_ranking_changed += 1
            if self.updated_ranking_changed >= 2:
                return finish
        if self.verbose:
            print(sor_grade, val, i)
        self.updated_ranking[sor_grade] = val
        grades = list(reversed(sorted([s for s in self.get_scores(i)])))
        if abs(grades[0] - grades[1]) < 0.01 or grades[0] < 0.3:
            return finish

    def get_scores(self, _i):
        l = self.x[_i]
        if self.with_scaling:
            l = [x * self.accuracy for x in l]
        for k in self.updated_ranking.keys():
            val = self.updated_ranking[k]
            l[k] = val
        return l


def get_x_y(feature: BaseParams, verbose, train_vs_test, with_scaling=False):
    from features.base_features import read_one_feature

    _, x_train, y_train = read_one_feature(feature, train_vs_test=train_vs_test, only_ab=False, to_binary=False)
    return Agent(x_train, y_train, feature, verbose=verbose, with_scaling=with_scaling)


class Game:
    def __init__(self, agents: List[Agent], verbose):
        self.agents = agents
        self.removed_from_round: List[Agent] = []
        self.verbose = verbose
        self._i = 0
        self.last_winner = None

    def reset(self):
        self.removed_from_round = []
        self.last_winner = None
        for a in self.agents:
            a.reset()

    def upp_i(self):
        self._i += 1
        self.reset()

    def get_remaining_agents(self):
        remaining_agents = []
        for agent in self.agents:
            if agent in self.removed_from_round:
                continue
            else:
                remaining_agents.append(agent)
        return remaining_agents

    def sort_by_highest(self) -> List[Agent]:
        agents = reversed(sorted([a for a in self.get_remaining_agents()], key=lambda a: a.find_highest_prob(self._i)))
        return list(agents)

    def find_two_highest(self) -> Tuple[Tuple[Agent, int], Tuple[Agent, int]]:
        agents = self.sort_by_highest()
        agent = agents[0]
        sor = agent.find_highest_sor_grade(self._i)

        opponent, opponent_sor = None, None
        for i in range(1, len(agents)):
            opponent = agents[i]
            opponent_sor = opponent.find_highest_sor_grade(self._i)
            if opponent_sor != sor:
                break

        if opponent_sor == sor:
            opponent, opponent_sor = None, None

        if self.verbose:
            print("agent's SOR:\t", agent.get_scores(self._i), sor, 'TRUE:', agent.y[self._i], agent.feature_name)
            if opponent is None:
                print("oponent's SOR:\t", None)
            else:
                print("oponent's SOR:\t", opponent.get_scores(self._i), opponent_sor, 'TRUE:', opponent.y[self._i],
                      opponent.feature_name)
        return (agent, sor), (opponent, opponent_sor)

    def small_round(self):
        self.sort_by_highest()
        (FA, FAU), (SA, SAU) = self.find_two_highest()
        if SA is None:
            return FAU
        FA_keep = FA.get_rank(FAU, self._i) - FA.get_rank(SAU, self._i)
        SA_keep = SA.get_rank(SAU, self._i) - SA.get_rank(FAU, self._i)
        FA_change = (FA.get_rank(FAU, self._i) + FA.get_rank(SAU, self._i)) / 2
        SA_change = (SA.get_rank(SAU, self._i) + SA.get_rank(FAU, self._i)) / 2

        if self.verbose:
            from tabulate import tabulate
            table = [
                ["", "SA_keep", "SA_change"],
                ["FA_keep", (FA_keep, SA_keep), (FA_keep, SA_change)],
                ["FA_change", (FA_change, SA_keep), (FA_change, SA_change)],
            ]
            print(tabulate(table, floatfmt=".2f"))

            table = [
                ["", "SA_keep", "SA_change"],
                ["FA_keep", (FA_keep + SA_keep), (FA_keep + SA_change)],
                ["FA_change", (FA_change + SA_keep), (FA_change + SA_change)],
            ]
            print(tabulate(table, floatfmt=".2f"))

        FA_action = keep if FA_keep > FA_change else (any_ if FA_keep == FA_change else change)
        SA_action = keep if SA_keep > SA_change else (any_ if SA_keep == SA_change else change)
        if self.verbose:
            print("FA", FA_action)
            print("SA", SA_action)
            print('FA_keep', FA_keep)
            print('SA_keep', SA_keep)
            print('FA_change', FA_change)
            print('SA_change', SA_change)

        if FA_action in [keep, any_] and SA_action in [change, any_]:
            self.last_winner = FA
            self.removed_from_round.append(SA)
            if self.verbose:
                print(FA.feature_name, 'is winner')

            if FA.update_ranking(FAU, FA_keep, keep, self._i) == "SECOND TIME":
                self.removed_from_round.append(FA)

        elif SA_action in [keep, any_] and FA_action in [change, any_]:
            self.last_winner = SA
            self.removed_from_round.append(FA)
            if self.verbose:
                print(SA.feature_name, 'is winner')

            if SA.update_ranking(SAU, SA_keep, keep, self._i) == "SECOND TIME":
                self.removed_from_round.append(SA)

        elif FA_action == keep and SA_action == keep:
            self.last_winner = FA
            if self.verbose:
                print(FA.feature_name, 'is winner')
            if FA.update_ranking(FAU, FA_keep, keep, self._i) == "SECOND TIME":
                self.removed_from_round.append(FA)
            if SA.update_ranking(SAU, SA_keep, keep, self._i) == "SECOND TIME":
                self.removed_from_round.append(SA)
        elif FA_action == change and SA_action == change:
            self.last_winner = FA
            if self.verbose:
                print(FA.feature_name, 'is winner')
            if FA.update_ranking(FAU, FA_change, change, self._i) == "SECOND TIME":
                self.removed_from_round.append(FA)
            if SA.update_ranking(SAU, SA_change, change, self._i) == "SECOND TIME":
                self.removed_from_round.append(SA)

    def get_winner(self) -> int:
        if len(self.sort_by_highest()) > 0:
            a = self.sort_by_highest()[0]
        else:
            a = self.last_winner
        return a.find_highest_sor_grade(self._i)

    def play(self):
        predictions = []
        for i in range(len(self.agents[0].x)):
            if self.verbose:
                print("\n----------------------")
                print("round:", i)
                print("----------------------\n")
            winner: Optional[int] = None
            while len(self.agents) > len(self.removed_from_round) + 1:
                round = self.small_round()
                if round is not None:
                    winner = round
                    break
            if not winner:
                winner = self.get_winner()
            if self.verbose:
                print(winner, "TRUE:", self.agents[0].y[i])
            predictions.append(winner)
            self.upp_i()
        y = self.agents[0].y
        from utils.metrics import calculate_metrics
        return calculate_metrics(y, predictions)


def create_game_from_feature_names(feature_names, verbose=False, mode="test", with_scaling=False):
    agents = [get_x_y(feature_name, verbose, mode, with_scaling) for i, feature_name in
              enumerate(feature_names)]
    return Game(agents, verbose)


def run(features: List[BaseParams], verbose=False, with_save=False, with_scaling=False):
    game = create_game_from_feature_names(features, verbose, mode="train", with_scaling=with_scaling)
    m_train = game.play()

    game = create_game_from_feature_names(features, verbose, mode="test", with_scaling=with_scaling)
    m_test = game.play()

    new_fnames = [fn.get_features_names_str() for fn in features]
    feature_names_str = ', '.join(new_fnames)

    name = game_theory_name + '_on_new'
    params = StackedParams(features=features, classifier="game_theory", dirname=name,
                           params={'with_scaling': with_scaling}, pca=False, use_only_ab=False, to_binary=False)
    r = Result(metrics=m_test, params=params)
    print(m_test.to_string())
    if with_save:
        r.save_result()
    return m_test.accuracy, (feature_names_str, m_train.accuracy * 100, m_test.accuracy * 100)
