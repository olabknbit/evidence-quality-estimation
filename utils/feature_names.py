from typing import List

abstract_key = 'abstract'
abstract_most_replaced_key = 'abstract_most_replaced'
abstract_disease_or_syndrome_replaced_key = 'abstract_disease_or_syndrome_replaced'

conclusions_key = 'conclusions'
conclusions_most_replaced_key = 'conclusions_most_replaced'
conclusions_disease_or_syndrome_replaced_key = 'conclusions_disease_or_syndrome_replaced'

is_structured_key = 'is_structured'
journal_title_key = 'journal_title'

mesh_headings_key = 'mesh_headings'

methods_key = 'methods'
methods_most_replaced_key = 'methods_most_replaced'
methods_disease_or_syndrome_replaced_key = 'methods_disease_or_syndrome_replaced'
# methods_and_conclusions_key = 'methods_and_conclusions'

publication_type_key = 'publication_type'
publication_year_key = 'publication_year'

title_key = 'title'
title_disease_or_syndrome_replaced_key = 'title_disease_or_syndrome_replaced'
title_most_replaced_key = 'title_most_replaced'

all_feature_names = [
    abstract_key,
    abstract_disease_or_syndrome_replaced_key,
    abstract_most_replaced_key,

    conclusions_key,
    conclusions_disease_or_syndrome_replaced_key,
    conclusions_most_replaced_key,

    journal_title_key,

    mesh_headings_key,

    methods_key,
    methods_disease_or_syndrome_replaced_key,
    methods_most_replaced_key,

    publication_type_key,

    publication_year_key,

    title_key,
    title_disease_or_syndrome_replaced_key,
    title_most_replaced_key,
]


def max_one_of(l1: List[str], l2: List[str]) -> bool:
    count = 0
    for i1 in l1:
        for i2 in l2:
            if i1 == i2:
                count += 1
    return count <= 1


def max_one_title(l: List[str]) -> bool:
    return max_one_of(l, [title_disease_or_syndrome_replaced_key, title_key, title_most_replaced_key])


abstracts_list = [abstract_disease_or_syndrome_replaced_key, abstract_key, abstract_most_replaced_key]
conclusions_list = [conclusions_key, conclusions_disease_or_syndrome_replaced_key, conclusions_most_replaced_key]
methods_list = [methods_key, methods_disease_or_syndrome_replaced_key, methods_most_replaced_key]


def max_one_abstract(l: List[str]) -> bool:
    conc_list = abstracts_list + conclusions_list
    met_list = abstracts_list + methods_list
    return max_one_of(l, conc_list) and max_one_of(l, met_list)


def max_one_conclusions(l: List[str]) -> bool:
    conc_list = abstracts_list + conclusions_list
    return max_one_of(l, conc_list)


def max_one_methods(l: List[str]) -> bool:
    conc_list = abstracts_list + methods_list
    return max_one_of(l, conc_list)


def must_include_one_of(l: List[str]):
    must = [
        abstract_disease_or_syndrome_replaced_key,
        abstract_most_replaced_key,
        title_most_replaced_key,
    ]

    for i in must:
        if i in l:
            return True
    return False


def must_include_all_of(l: List[str]):
    must = [
        publication_type_key,
    ]

    for i in must:
        if i not in l:
            return False
    return True


# base
balanced_sklearn_name = 'balanced_sklearn'
sklearn_name = 'sklearn'
sklearn_lemmatizer_name = "sklearn_lemmatizer"
sklearn_master_name = "sklearn_master"
sklearn_master_undersampling_name = "sklearn_master_undersampling"
sklearn_master_oversampling_name = "sklearn_master_oversampling"
sklearn_master_pt_name = "sklearn_master_pt"

base_dirnames = [
    balanced_sklearn_name,
    sklearn_name,
    sklearn_lemmatizer_name,
    sklearn_master_name,
    sklearn_master_undersampling_name,
    sklearn_master_oversampling_name,
    sklearn_master_pt_name
]

sklearn_name = "sklearn"
mixed_name = "mixed"

# Classifier names
AdaBoostClassifier_name = "AdaBoostClassifier"
BernouliNB_name = "BernouliNB"
ExtraTreesClassifier_name = "ExtraTreesClassifier"
ExtraTreesRegressor_name = "ExtraTreesRegressor"
DecisionTreeClassifier_name = "DecisionTreeClassifier"
GaussianNB_name = "GaussianNB"
GaussianProcessClassifier_name = "GaussianProcessClassifier"
GradientBoostingClassifier_name = "GradientBoostingClassifier"
KNeighborsClassifier_name = "KNeighborsClassifier"
LinearSVC_name = "LinearSVC"
MLPClassifier_name = "MLPClassifier"
MLPRegressor_name = "MLPRegressor"
RadiusNeighborsClassifier_name = "RadiusNeighborsClassifier"
RandomForestClassifier_name = "RandomForestClassifier"
RandomForestClassifier500_name = "RandomForestClassifier500"
RandomForestRegressor_name = "RandomForestRegressor"
SGDClassifier_name = "SGDClassifier"
StackingClassifier_name = "StackingClassifier"
SVC_name = "SVC"
SVC_with_linear_kernel_name = "SVC_with_linear_kernel"
VotingClassifier_name = "VotingClassifier"


def get_all_feature_subsets() -> List[List[str]]:
    from utils.stacking import get_all_subsets

    all_subsets = get_all_subsets(all_feature_names)
    all_subsets = list(filter(lambda x: len(x) >= 2, all_subsets))
    all_subsets = list(filter(max_one_title, all_subsets))
    all_subsets = list(filter(max_one_abstract, all_subsets))
    all_subsets = list(filter(max_one_methods, all_subsets))
    all_subsets = list(filter(max_one_conclusions, all_subsets))
    # all_subsets = list(filter(must_include_one_of, all_subsets))
    all_subsets = list(filter(must_include_all_of, all_subsets))

    return all_subsets
