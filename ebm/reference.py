import os.path
from typing import List, Optional
from xml.etree.ElementTree import Element


def print_elem_tree_for_debug(path: str, root: Element) -> None:
    print(path)
    for child in root:
        print(child)
        for child2 in child:
            print(' ', child2)
            for child3 in child2:
                print('  ', child3)
                for child4 in child3:
                    print('    ', child4)


def get_metamapped_content(dirname: str, reference_id: str):
    from ebm.filenames import metamapped_dir
    filepath = os.path.join(metamapped_dir, dirname, reference_id + '.txt')
    with open(filepath, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        if len(lines) == 0:
            lines = [""]
        return lines[0]


class Reference:
    def __init__(self, ref_id: str, title: str, journal_title: str, abstract: str, is_structured: bool, background: str,
                 objective: str, methods: str, results: str, conclusions: str, publication_year: str,
                 mesh_headings_majortopic_y: List[str], mesh_headings_majortopic_n: List[str],
                 publication_types: List[str]):
        self.id: str = ref_id
        self.title: str = title
        self.journal_title: str = journal_title
        self.abstract: str = abstract
        self.publication_types: List[str] = publication_types
        self.publication_year: Optional[str] = publication_year
        self.mesh_headings_majortopic_y: List[str] = mesh_headings_majortopic_y
        self.mesh_headings_majortopic_n: List[str] = mesh_headings_majortopic_n
        # Abstract sections ['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS']
        self.background = background
        self.objective = objective
        self.methods = methods
        self.methods = self.get_methods()
        self.results = results
        self.conclusions = conclusions
        self.conclusions = self.get_conclusions()
        self.is_structured = is_structured

        self.preferred_title: Optional[str] = None
        self.title_disease_or_syndrome_replaced: Optional[str] = None
        self.title_most_replaced: Optional[str] = None
        self.title_all_replaced: Optional[str] = None

        self.abstract_most_replaced: Optional[str] = None
        self.abstract_disease_or_syndrome_replaced: Optional[str] = None

        self.conclusions_most_replaced: Optional[str] = None
        self.conclusions_disease_or_syndrome_replaced: Optional[str] = None

        self.methods_most_replaced: Optional[str] = None
        self.methods_disease_or_syndrome_replaced: Optional[str] = None

    # Returns a concatenation of a title and an abstract.
    def to_line(self):
        return self.title + ' ' + self.abstract

    def get_methods(self) -> str:
        if self.methods is None:
            from features.extract_methods_and_conclusions_sections import extract_method_from_abstract
            self.methods = extract_method_from_abstract(self.abstract)
        return self.methods

    def get_conclusions(self) -> str:
        if self.conclusions is None:
            from features.extract_methods_and_conclusions_sections import extract_conclusion_from_abstract
            self.conclusions = extract_conclusion_from_abstract(self.abstract)
        return self.conclusions

    def fill_preferred_title(self):
        self.preferred_title = get_metamapped_content('title_preferred', self.id)

    def fill_title_disease_or_syndrome_replaced(self):
        self.title_disease_or_syndrome_replaced = get_metamapped_content('title_disease_or_syndrome_replaced', self.id)

    def fill_title_most_replaced(self):
        self.title_most_replaced = get_metamapped_content('title_most_replaced', self.id)

    def fill_title_all_replaced(self):
        self.title_all_replaced = get_metamapped_content('title_all_replaced', self.id)

    def fill_abstract_most_replaced(self):
        self.abstract_most_replaced = get_metamapped_content('abstract_most_replaced', self.id)

    def fill_abstract_disease_or_syndrome_replaced(self):
        self.abstract_disease_or_syndrome_replaced = get_metamapped_content('abstract_disease_or_syndrome_replaced',
                                                                            self.id)

    def fill_conclusions_most_replaced(self):
        self.conclusions_most_replaced = get_metamapped_content('conclusions_most_replaced', self.id)

    def fill_conclusions_disease_or_syndrome_replaced(self):
        self.conclusions_disease_or_syndrome_replaced = get_metamapped_content(
            'conclusions_disease_or_syndrome_replaced', self.id)

    def fill_methods_most_replaced(self):
        self.methods_most_replaced = get_metamapped_content('methods_most_replaced', self.id)

    def fill_methods_disease_or_syndrome_replaced(self):
        self.methods_disease_or_syndrome_replaced = get_metamapped_content('methods_disease_or_syndrome_replaced',
                                                                           self.id)

    def fill_all_metamapped(self):
        self.fill_abstract_disease_or_syndrome_replaced()
        self.fill_abstract_most_replaced()
        self.fill_title_disease_or_syndrome_replaced()
        self.fill_title_most_replaced()
        self.fill_methods_disease_or_syndrome_replaced()
        self.fill_methods_most_replaced()
        self.fill_conclusions_disease_or_syndrome_replaced()
        self.fill_conclusions_most_replaced()


def parse_reference_document(ref_id: str, root: Element):
    from ebm import parse_reference
    reference = None
    if root.find('pubmedarticle') is not None \
            or root.find('PubmedArticle') is not None \
            or root.find('pubmedArticle') is not None:

        article = parse_reference.get_article(root)
        title = parse_reference.get_title(article)
        journal_title = parse_reference.get_journal_title(article)
        publication_types = parse_reference.get_publication_types(article)

        abstract = parse_reference.get_abstract_text(article)
        if abstract == '':
            abstract = parse_reference.get_abstract_text_from_medline(root)
        is_structured = parse_reference.get_is_structured(article)
        background = parse_reference.get_abstract_section(article, 'BACKGROUND')
        objective = parse_reference.get_abstract_section(article, 'OBJECTIVE')
        methods = parse_reference.get_abstract_section(article, 'METHODS')
        results = parse_reference.get_abstract_section(article, 'RESULTS')
        conclusions = parse_reference.get_abstract_section(article, 'CONCLUSIONS')

        publication_year = parse_reference.get_publication_year_for_article(root)
        mesh_headings_majortopic_y, mesh_headings_majortopic_n = parse_reference.get_mesh_headings(root)
        publication_types = [pub_type.text.strip() for pub_type in publication_types]
        reference = Reference(ref_id, title, journal_title, abstract, is_structured, background, objective, methods,
                              results,
                              conclusions, publication_year, mesh_headings_majortopic_y, mesh_headings_majortopic_n,
                              publication_types)

    elif root.find('pubmedbookarticle') is not None:
        pubmed_book_article = root.find('pubmedbookarticle')
        bookdocument = pubmed_book_article.find('bookdocument')
        book = bookdocument.find('book')

        title = parse_reference.get_title(book)
        journal_title = parse_reference.get_book_title(book)
        publication_types = parse_reference.get_publication_types(bookdocument)

        abstract = parse_reference.get_abstract_text(bookdocument)

        is_structured = parse_reference.get_is_structured(bookdocument)
        background = parse_reference.get_abstract_section(bookdocument, 'BACKGROUND')
        objective = parse_reference.get_abstract_section(bookdocument, 'OBJECTIVE')
        methods = parse_reference.get_abstract_section(bookdocument, 'METHODS')
        results = parse_reference.get_abstract_section(bookdocument, 'RESULTS')
        conclusions = parse_reference.get_abstract_section(bookdocument, 'CONCLUSIONS')

        publication_year = parse_reference.get_publication_year_for_book(book)
        mesh_headings_majortopic_y, mesh_headings_majortopic_n = parse_reference.get_mesh_headings(root)
        publication_types = [pub_type.text.strip() for pub_type in publication_types]
        reference = Reference(ref_id, title, journal_title, abstract, is_structured, background, objective, methods,
                              results,
                              conclusions, publication_year, mesh_headings_majortopic_y, mesh_headings_majortopic_n,
                              publication_types)
    reference.fill_all_metamapped()
    return reference


# Retrieves the abstract from 'Abstracts' dir
def create_reference(ref_id: str, directory: str) -> Optional[Reference]:
    import os.path
    path = os.path.join(directory, ref_id + '.xml')
    if not os.path.exists(path):
        print('Error', path, 'has no file')
        return
    with open(path, 'r', encoding="utf-8") as f:
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(f).getroot()

            reference = parse_reference_document(ref_id, root)
            if reference.abstract == '':
                return None
            else:
                return reference

        except Exception:
            print_elem_tree_for_debug(path, root)
            import traceback

            traceback.print_exc()
