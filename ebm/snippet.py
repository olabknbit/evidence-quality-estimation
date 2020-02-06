import xml.etree.ElementTree as ET
from typing import Optional, List

from ebm.reference import Reference, create_reference


class Snippet:
    def __init__(self, evidence_id: str, sniptext: str, sor_grade: str, references: List[Reference], question: str,
                 sor_text: str = ''):
        self.evidence_id = evidence_id
        self.sniptext = sniptext
        self.sor_grade = sor_grade
        self.references: List[Reference] = references
        self.question = question
        self.sor_text = sor_text

    def get_features(self, f):
        if f.__name__ == "publication_type" and self.sor_text != '' and self.sor_text:
            # print("ST", self.sor_text)
            from features.extract_publication_type import extract_from_text
            pub_types = extract_from_text(self.sor_text)
            # pub_type = pub_types[0] if len(pub_types) > 0 else "unknown"
            # print("NS", pub_type)
            X = pub_types
            # print("NS", pub_type)
        else:
            X = list(set([f(reference) for reference in self.references]))
            # print("FR", X)

        # X = list(set([f(reference) for reference in self.references]))
        return X


def create_snippet(record_id: str, snippet: ET.Element, question: str, abstracts_dir: str) -> Optional[Snippet]:
    sor = snippet.find('sor').attrib['type']
    if sor not in ['A', 'B', 'C']:
        return None

    snip_id = snippet.attrib['id']
    sniptext = snippet.find('sniptext').text
    longs = snippet.findall('long')
    # TODO these are the 'answers' in
    #  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.227.8941&rep=rep1&type=pdf !
    references = []

    for long in longs:
        if long is None:
            continue
        refs = long.findall('ref')
        for ref in refs:
            if ref is None:
                break
            ref_id = ref.attrib['id']
            if 'NOT_FOUND' in ref_id:
                continue
            reference = create_reference(ref_id, abstracts_dir)
            if reference:
                references.append(reference)

    if len(references) == 0:
        return None
    return Snippet(record_id + snip_id, sniptext, sor, references, question)
