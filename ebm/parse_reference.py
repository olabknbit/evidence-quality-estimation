from typing import List, Optional, Set, Tuple
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


# The input must have all letters that can possibly be capitalized capitalized on input.
def generate_all_variations_of_word(word: str, words: Set[str]) -> Set[str]:
    words.add(word)
    for i in range(len(word)):
        if word[i].isupper():
            new_word = list(word)
            new_word[i] = new_word[i].lower()
            new_word = ''.join(new_word)
            words.update(generate_all_variations_of_word(new_word, words))
    return words


def get_element_with_name(root: Element, names: List[str]) -> Optional[Element]:
    for cap_name in names:
        for name in generate_all_variations_of_word(cap_name, set()):
            if root.find(name) is not None:
                return root.find(name)

    return None


def get_all_elems_with_name(root: Element, names: List[str]) -> List[Element]:
    for cap_name in names:
        for name in generate_all_variations_of_word(cap_name, set()):
            if len(root.findall(name)) > 0:
                return root.findall(name)

    return []


def get_abstract_elems(article: Element) -> List[Element]:
    abstract = get_element_with_name(article, ['Abstract', 'OtherAbstract'])
    if abstract is None:
        return []
    return get_all_elems_with_name(abstract, ['AbstractText'])


def get_abstract_text(article: Element) -> str:
    abstract_texts = get_abstract_elems(article)
    return ' '.join([abstract_t.text.strip() for abstract_t in abstract_texts])


def get_is_structured(article: Element) -> bool:
    abstract_texts = get_abstract_elems(article)
    return len(abstract_texts) > 1


def get_abstract_section(article: Element, section_name: str) -> Optional[str]:
    abstract_texts = get_abstract_elems(article)
    for elem in abstract_texts:
        if 'nlmcategory' in elem.attrib and elem.attrib['nlmcategory'] == section_name:
            return elem.text
    return None


def get_publication_types(article: Element) -> List[Element]:
    publication_type_list = get_element_with_name(article, ['PublicationTypeList'])
    if publication_type_list is None:
        return []
    return get_all_elems_with_name(
        publication_type_list, ['PublicationType'])


def get_medline_citation(root: Element) -> Optional[Element]:
    pubmed_article = get_element_with_name(root, ['PubmedArticle', 'PubmedBookArticle'])
    return get_element_with_name(pubmed_article, ['MedlineCitation', 'BookDocument'])


def get_article(root: Element) -> Optional[Element]:
    medline_citation = get_medline_citation(root)
    return get_element_with_name(medline_citation, ['Article', 'Book'])


def get_abstract_text_from_medline(root: Element) -> str:
    medline_citation = get_medline_citation(root)
    return get_abstract_text(medline_citation)


def get_publication_year_for_article(root: Element) -> Optional[str]:
    medline_citation = get_medline_citation(root)
    date_elem = get_element_with_name(medline_citation, ['DateCompleted', 'DateCreated'])
    if date_elem is None:
        return None

    return get_element_with_name(date_elem, ['Year']).text.strip()


def get_publication_year_for_book(book: Element) -> Optional[str]:
    date_elem = get_element_with_name(book, ['PubDate'])

    if date_elem is None:
        return None

    return get_element_with_name(date_elem, ['Year']).text.strip()


def get_mesh_headings(root: Element) -> Tuple[List[str], List[str]]:
    medline_citation = get_medline_citation(root)
    mesh_heading_list = get_element_with_name(medline_citation, ['MeSHHeadingList'])
    if mesh_heading_list is None:
        return [], []
    else:
        mesh_headings = get_all_elems_with_name(mesh_heading_list, ['MeSHHeading'])
        desc_names = [get_element_with_name(e, ['DescriptorName']) for e in mesh_headings]
        for e in desc_names:
            if 'majortopicyn' not in e.attrib and 'MajorTopicYN' not in e.attrib:
                print(e.attrib)
                exit(1)
        majortopic_yn = [e.attrib['majortopicyn'] if 'majortopicyn' in e.attrib else e.attrib['MajorTopicYN'] for e in
                         desc_names]
        mesh_headings = [e.text.strip() for e in desc_names]
        y, n = [], []
        for yn, mesh_heading in zip(majortopic_yn, mesh_headings):
            if yn == 'N':
                n.append(mesh_heading)
            else:
                y.append(mesh_heading)

        return y, n


def get_title(article: Element) -> str:
    return get_element_with_name(article, ['ArticleTitle', 'BookTitle']).text.strip()


def get_journal_title(article: Element) -> str:
    journal = get_element_with_name(article, ['Journal'])
    if journal is not None:
        return get_element_with_name(journal, ['Title']).text.strip()


def get_book_title(article: Element) -> str:
    return get_element_with_name(article, ['BookTitle', 'ArticleTitle']).text.strip()
