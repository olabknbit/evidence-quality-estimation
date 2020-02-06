from typing import List, Callable

from ebm.reference import Reference


def abstract(reference: Reference) -> str:
    return reference.abstract


def methods(reference: Reference) -> str:
    return reference.get_methods()


def conclusions(reference: Reference) -> str:
    return reference.get_conclusions()


def title(reference: Reference) -> str:
    return reference.title


def mesh_headings(reference: Reference) -> str:
    return ' '.join(reference.mesh_headings_majortopic_y)


def journal_title(reference: Reference) -> str:
    return reference.journal_title


def publication_type(reference: Reference) -> str:
    from features.extract_publication_type import new_method
    return new_method(reference)


def publication_year(reference: Reference) -> str:
    return str(reference.publication_year)


def preferred_title(reference: Reference) -> str:
    return reference.preferred_title


def title_disease_or_syndrome_replaced(reference: Reference) -> str:
    return reference.title_disease_or_syndrome_replaced


def title_most_replaced(reference: Reference) -> str:
    return reference.title_most_replaced


def title_all_replaced(reference: Reference) -> str:
    return reference.title_all_replaced


def abstract_most_replaced(reference: Reference) -> str:
    return reference.abstract_most_replaced


def abstract_disease_or_syndrome_replaced(reference: Reference) -> str:
    return reference.abstract_disease_or_syndrome_replaced


def conclusions_most_replaced(reference: Reference) -> str:
    return reference.conclusions_most_replaced


def conclusions_disease_or_syndrome_replaced(reference: Reference) -> str:
    return reference.conclusions_disease_or_syndrome_replaced


def methods_most_replaced(reference: Reference) -> str:
    return reference.methods_most_replaced


def methods_disease_or_syndrome_replaced(reference: Reference) -> str:
    return reference.methods_disease_or_syndrome_replaced


funs = [abstract,
        abstract_disease_or_syndrome_replaced,
        abstract_most_replaced,
        conclusions,
        conclusions_disease_or_syndrome_replaced,
        conclusions_most_replaced,
        journal_title,
        mesh_headings,
        methods,
        methods_disease_or_syndrome_replaced,
        methods_most_replaced,
        publication_type,
        publication_year,
        preferred_title,
        title,
        title_all_replaced,
        title_disease_or_syndrome_replaced,
        title_most_replaced,
        ]


def get_f(feature_name) -> Callable[[Reference], str]:
    names = [f.__name__ for f in funs]
    if feature_name not in names:
        print("No such feature function:", feature_name, "available functions: ", names)
        exit(1)

    for fun in funs:
        if feature_name == fun.__name__:
            return fun


def get_fs(feature_names) -> List[Callable[[Reference], str]]:
    names = [f.__name__ for f in funs]
    result = []
    print(feature_names)
    for feature_name in feature_names:
        if feature_name not in names:
            print("No such feature function:", feature_name, "available functions: ", names)
            exit(1)

        for fun in funs:
            if feature_name == fun.__name__:
                result.append(fun)
    return result
