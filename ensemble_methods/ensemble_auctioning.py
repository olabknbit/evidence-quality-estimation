from typing import Optional, List

from params import StackedParams, BaseParams


class Agent:
    def __init__(self, x: List[List[float]], y: List[str], feature: BaseParams, i: int):
        self.x = x
        from utils.mapping import map_y_array_to_int
        self.y = map_y_array_to_int(y)
        self.feature = feature
        self.feature_name = feature.get_features_names_str()
        self.loses = 0
        self.id = i

    def get_chosen_class(self, i):
        values = self.x[i]
        return values.index(max(values))

    def update_ranking(self, val, i):
        sor_grade = self.get_chosen_class(i)
        self.x[i][sor_grade] -= val

    def compute_cost(self, other_sor, i):
        sor_grade = self.get_chosen_class(i)
        return self.x[i][sor_grade] - self.x[i][other_sor]

    def get_highest_confidence_score(self, i):
        return max(self.x[i])


def get_x_y(feature: BaseParams, i):
    from features.base_features import read_one_feature
    _, x_train, y_train = read_one_feature(feature, train_vs_test="train", only_ab=False, to_binary=False)

    return Agent(x_train, y_train, feature, i)


class Game:
    def __init__(self, agents: List[Agent], C: float, verbose):
        self.agents = agents
        self.removed_from_round = []
        self.C: float = C
        self.last_winner: Optional[Agent] = None
        self.verbose = verbose

    def get_all_playing_agents(self):
        return [agent for i, agent in enumerate(self.agents) if i not in self.removed_from_round]

    def small_round(self, i):
        playing_agents = self.get_all_playing_agents()
        if self.verbose:
            print("still playing:", len(playing_agents))
        accumulated_cost = []
        max_confidence = 0.
        for playing_agent in playing_agents:
            ag_conf_score = playing_agent.get_highest_confidence_score(i)
            if ag_conf_score > max_confidence:
                max_confidence = ag_conf_score
                self.last_winner = playing_agent
            if self.verbose:
                print("agent", playing_agent.id, playing_agent.x[i])
            agent_cost = 0
            for rel_playing_agent in playing_agents:
                if playing_agent == rel_playing_agent:
                    continue

                rel_ag_sor = rel_playing_agent.get_chosen_class(i)
                # print("opponent", rel_playing_agent.id, rel_playing_agent.x[i], "sor:", rel_ag_sor)
                rel_ag_cost = playing_agent.compute_cost(rel_ag_sor, i)
                # print("rel_ag_cost", rel_ag_cost)
                agent_cost += rel_ag_cost
            agent_cost /= self.C
            if self.verbose:
                print("agent", playing_agent.id, "overall_cost", agent_cost)
            accumulated_cost.append(agent_cost)
            playing_agent.update_ranking(agent_cost, i)
            if playing_agent.get_highest_confidence_score(i) < 0.2:
                if self.verbose:
                    print("removing agent")
                self.removed_from_round.append(playing_agent.id)
        loser_index = accumulated_cost.index(max(accumulated_cost))
        loser = playing_agents[loser_index]
        loser.loses += 1
        if loser.loses == 2:
            self.removed_from_round.append(loser.id)

    def get_winner(self, i):
        for i, agent in enumerate(self.agents):
            if i in self.removed_from_round:
                continue
            return agent.get_chosen_class(i)
        return self.last_winner.get_chosen_class(i)

    def reset(self):
        self.removed_from_round = []
        for a in self.agents:
            a.loses = 0

    def play(self):
        predictions = []
        for i in range(len(self.agents[0].x)):
            if self.verbose:
                print("\n----------------------")
                print("round:", i, "SOR: ", self.agents[0].y[i])
                print("----------------------\n")

            while len(self.agents) > len(self.removed_from_round) + 1:
                self.small_round(i)

            winner = self.get_winner(i)
            if self.verbose:
                print(winner, "TRUE:", self.agents[0].y[i])

            predictions.append(winner)
            self.reset()
        y = self.agents[0].y
        from utils.metrics import calculate_metrics
        m = calculate_metrics(y, predictions)
        if self.verbose:
            print(m.accuracy, m.aed_score, self.C)

        return m


def create_game(features: List[BaseParams], x, verbose=False):
    agents = [get_x_y(feature, i) for i, feature in
              enumerate(features)]
    return Game(agents, x, verbose)


def run(features: List[BaseParams], verbose, with_save, name):
    max_acc = 0.
    max_x = 0.
    min_aed = 100.
    min_x = 0.
    metrics = None
    import numpy as np
    for x in np.arange(2, 8, 0.1):
        game = create_game(features, x, verbose=verbose)
        m = game.play()
        if m.accuracy > max_acc:
            max_acc = m.accuracy
            max_x = x
            metrics = m
        if m.aed_score < min_aed:
            min_aed = m.aed_score
            min_x = x
    print("max_acc", max_acc, max_x)
    print("min_aed", min_aed, min_x)
    from utils.metrics import Result
    from ebm.filenames import auctioning_name
    name = auctioning_name + '_on_' + name
    r = Result(metrics=metrics,
               params=StackedParams(classifier=name, dirname=name, features=features, params={}, to_binary=False,
                                    use_only_ab=False, pca=False))
    print(r.metrics.to_string())
    if with_save:
        r.save_result()
    return max_acc
