from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt

from utils.metrics import read_test_result_from_metrics_file, Result, filter_only_metrics_filenames


class TrainerAccuracies:
    def __init__(self, metric="balanced_accuracy_adjusted"):
        self.d: Dict[str, List[Result]] = {}
        self.metric = metric

    def get_method_trainer(self, result: Result):
        return result.dirname + '.' + result.classifier

    def parse_filenames(self, paths: List[str]) -> None:
        """
        Parses filenames to get results
        :param paths:
        :return: Dict with key: feature_type and value: list of Results
        """
        for path in paths:
            result = read_test_result_from_metrics_file(path=path)
            self.d[self.get_method_trainer(result)] = self.d.get(self.get_method_trainer(result), [])
            self.d[self.get_method_trainer(result)].append(result)

    def plot(self) -> None:
        import matplotlib._color_data as mcd
        shape = 'o'
        colors = mcd.XKCD_COLORS
        # ['#A93226', '#CB4335', '#884EA0', '#7D3C98', '#2471A3', '#2E86C1', '#17A589', '#138D75', '#229954',
        #       '#28B463', '#D4AC0D', '#D68910', '#CA6F1E', '#BA4A00']

        dictionary = self.d

        color_names = list(dictionary.keys())
        x_names = set()
        for color_name in color_names:
            for result in dictionary[color_name]:
                x_names.add(result.get_features_str())
        x_names = list(x_names)

        for color_name, color in zip(color_names, colors):
            results = dictionary[color_name]
            names = []
            values = []
            for result in results:
                names.append(result.get_features_str())
                values.append(getattr(result.metrics, self.metric))
            values = [x for _, x in sorted(zip(names, values))]
            names = [y for y, _ in sorted(zip(names, values))]

            plt.plot(names, values, marker=shape, color=color, linestyle='None')
        plt.xticks(x_names, x_names, rotation='vertical')

        plt.legend(color_names, loc='right', bbox_to_anchor=(1., 0.2))


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(3))
    plt.yticks([])
    thisplot = plt.bar(range(3), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_all(predictions, test_labels):
    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = 5
    num_cols = 5
    num_images = num_rows * num_cols
    plt.figure(figsize=(num_cols, num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()


def plot_features_filepaths(filepaths, metric="balanced_accuracy_adjusted") -> None:
    d = TrainerAccuracies(metric)
    d.parse_filenames(filter_only_metrics_filenames(filepaths))
    plt.figure(figsize=(10, 8))
    d.plot()
    plt.plot()


def plot_double_plot(filepaths, metric="balanced_accuracy_adjusted") -> None:
    d = TrainerAccuracies(metric)
    d.parse_filenames(filter_only_metrics_filenames(filepaths))

    d2 = TrainerAccuracies("accuracy")
    d2.parse_filenames(filter_only_metrics_filenames(filepaths))
    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = 1
    num_cols = 2
    plt.figure(figsize=(10, 8))

    plt.subplot(num_rows, num_cols, 1)
    d.plot()

    plt.subplot(num_rows, num_cols, 2)
    d2.plot()

    plt.tight_layout()
    plt.show()
