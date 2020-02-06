import os.path

from filename import path

data_dir = os.path.join(path, 'data')
ebm_sum_corpus_dir = os.path.join(data_dir, 'ebmsumcorpus')
abstracts_dir = os.path.join(ebm_sum_corpus_dir, 'Abstracts')
clinical_inquiries_xml = os.path.join(ebm_sum_corpus_dir, 'ClinicalInquiries.xml')

metamapped_dir = os.path.join(data_dir, 'metamapped')

alta_2011_shared_task_dir: str = os.path.join(data_dir, 'alta_2011_shared_task')
alta_2011_shared_task_dataset_dir = os.path.join(alta_2011_shared_task_dir, 'dataset')
alta_2011_shared_task_dataset_trainset_dir = os.path.join(alta_2011_shared_task_dataset_dir, 'trainset')
alta_2011_shared_task_dataset_trainset = os.path.join(alta_2011_shared_task_dataset_dir, 'trainset.txt')
alta_2011_shared_task_dataset_devtestset_dir = os.path.join(alta_2011_shared_task_dataset_dir, 'devtestset')
alta_2011_shared_task_dataset_devtestset = os.path.join(alta_2011_shared_task_dataset_dir, 'devtestset.txt')
alta_2011_shared_task_testset_dir = os.path.join(alta_2011_shared_task_dir, 'testset/')
alta_2011_shared_task_testset = os.path.join(alta_2011_shared_task_dir, 'testset.txt')

# base
balanced_sklearn_name = 'balanced_sklearn'
sklearn_name = 'sklearn'
sklearn_master_name = "sklearn_master"
sklearn_mixed_name = "sklearn_mixed"

base_keras_name = "base_keras"

# stacked
keras_name = "keras"
sklearn_stacked_name = "sklearn_stacked"
auctioning_name = 'auctioning'
game_theory_name = "game_theory"
consensus_name = "consensus"
bandit_name = "bandit"

out_dir = os.path.join(path, 'out')
metrics_dir = os.path.join(out_dir, 'metrics')
state_of_the_art_metrics_dir = os.path.join(metrics_dir, 'state_of_the_art')
sklearn_metrics_dir = os.path.join(metrics_dir, sklearn_name)
sklearn_master_metrics_dir = os.path.join(metrics_dir, sklearn_master_name)
sklearn_mixed_metrics_dir = os.path.join(metrics_dir, sklearn_mixed_name)

sklearn_stacked_metrics_dir = os.path.join(metrics_dir, sklearn_stacked_name)
keras_stacked_metrics_dir = os.path.join(metrics_dir, keras_name)
game_theory_metrics_dir = os.path.join(metrics_dir, game_theory_name)
auctioning_metrics_dir = os.path.join(metrics_dir, auctioning_name)
consensus_metrics_dir = os.path.join(metrics_dir, consensus_name)


def get_features_dir(feature_name: str, dirname: str, train_vs_test: str, classifier_name: str) -> str:
    return os.path.join(data_dir, dirname, train_vs_test, feature_name, classifier_name)


def get_metrics_path(feature_name: str, trainer: str, mode: str) -> str:
    return os.path.join(metrics_dir, mode, trainer, feature_name + '_' + 'metrics.txt')


def get_metrics_dir(mode: str) -> str:
    return os.path.join(metrics_dir, mode)
