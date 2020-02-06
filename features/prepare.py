# Functions in this module,
# given: Data
# return: tuple of evidence_ids, X, y <-> Tuple[List[str], List[List[str]], List[str]]

from typing import Callable, List, Tuple

from ebm.dataset import Data
from ebm.reference import Reference


def __features(data: Data, f: Callable[[Reference], str]) -> Tuple[List[str], List[List[str]], List[str]]:
    evidence_ids, X, y = [], [], []
    for snippet in data.snippets:
        evidence_id = snippet.evidence_id
        sor = snippet.sor_grade
        features = []
        for reference in snippet.references:
            features.append(f(reference))
        evidence_ids.append(evidence_id)
        X.append(features)
        y.append(sor)
    return evidence_ids, X, y


def methods_and_conclusion(data: Data) -> Tuple[List[str], List[List[str]], List[str]]:
    return __features(data, lambda ref: ref.get_both())


def methods(data: Data) -> Tuple[List[str], List[List[str]], List[str]]:
    return __features(data, lambda ref: ref.get_methods())


def conclusions(data: Data) -> Tuple[List[str], List[List[str]], List[str]]:
    return __features(data, lambda ref: ref.get_conclusions())


def publication_year(data: Data) -> Tuple[List[str], List[List[str]], List[str]]:
    return __features(data, lambda ref: ref.publication_year if ref.publication_year else '')


def mesh_headings(data: Data) -> Tuple[List[str], List[List[str]], List[str]]:
    return __features(data, lambda ref: ' '.join(ref.mesh_headings_majortopic_y))


def publication_type(data: Data) -> \
        Tuple[List[str], List[List[str]], List[str]]:
    from features.extract_publication_type import new_method
    return __features(data, new_method)


def abstract(data: Data) -> Tuple[List[str], List[List[str]], List[str]]:
    return __features(data, lambda ref: ref.abstract)


def title(data: Data) -> Tuple[List[str], List[List[str]], List[str]]:
    return __features(data, lambda ref: ref.title)


def journal_title(data: Data) -> Tuple[List[str], List[List[str]], List[str]]:
    return __features(data, lambda ref: ref.journal_title)


def abstract_title(data: Data) -> (List[str], List[List[str]], List[str]):
    return __features(data, lambda ref: ref.to_line())


def is_structured(data: Data) -> (List[str], List[List[str]], List[str]):
    return __features(data, lambda ref: "1" if ref.is_structured else "0")


def abstract_title_question(data: Data) -> (List[str], List[List[str]], List[str]):
    evidence_ids, X, y = [], [], []
    for snippet in data.snippets:
        evidence_id = snippet.evidence_id
        sor = snippet.sor_grade
        features = [snippet.question + '\n']
        for reference in snippet.references:
            features.append(reference.to_line())
        evidence_ids.append(evidence_id)
        X.append(features)
        y.append(sor)
    return evidence_ids, X, y
