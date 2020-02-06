import os.path
from typing import List, Optional, Callable

from ebm.dataset import get_all_original_data
from ebm.filenames import metamapped_dir
from ebm.reference import Reference
from metamap.metamap import Output
from utils.file_management import save_data_with_ultimate_dir_creation


def read_metamapped_output(reference_id, dirname):
    from metamap.metamap import get_metamap_output
    filepath = os.path.join(metamapped_dir, dirname, reference_id + '.txt')

    with open(filepath, 'r') as f:
        lines = f.readlines()
    text = '\n'.join(lines)
    return get_metamap_output(text)


def read_metamapped_title_output(reference: Reference) -> Output:
    return read_metamapped_output(reference.id, 'title_metamapped')


def read_metamapped_abstract_output(reference: Reference) -> Output:
    return read_metamapped_output(reference.id, 'abstract_metamapped')


def serialize_metamap(f: Callable[[Reference], str], dirname):
    from metamap.metamap import call_metamap
    import os.path
    data = get_all_original_data()
    references = data.get_all_references().items()
    count = 0
    for key, reference in references:
        print("progress", count, '/', len(references), ' : ', count / len(references), key)
        filepath = os.path.join(metamapped_dir, dirname, key + '.txt')
        if os.path.exists(filepath):
            print("skipping", key)
        else:
            output = call_metamap(f(reference))
            output = '\n'.join(output.split('\n')[14:])

            save_data_with_ultimate_dir_creation(filepath, [output])
        count += 1


def serialize_title_metamap():
    from features.get_from_reference import title
    serialize_metamap(title, "title_metamapped")


def serialize_abstract_metamap():
    from features.get_from_reference import abstract
    serialize_metamap(abstract, "abstract_metamapped")


def title_preferred(reference: Reference) -> str:
    print("original:\t", reference.title)

    op = read_metamapped_title_output(reference)
    text = reference.title
    for phrase in op.phrases:
        if len(phrase.meta_mappings) == 0:
            continue
        substitute_with = ' '.join(
            [mm.preferred_name if mm.preferred_name else mm.text for mm in phrase.meta_mappings[0]])
        text = text.replace(phrase.phrase, substitute_with)

    preferred = text
    print("preferred:\t", preferred)
    return preferred


def get_prefix(phrase: Optional[str], l: List[str]) -> Optional[str]:
    if not phrase:
        return None
    for i in l:
        if phrase.startswith(i):
            return i
    return None


def generic_replaced(op: Output, text: str, substitutions_list: List[str]) -> str:
    if not text:
        return ""
    print("original\t", text)
    for sentence, phrases in op.sentences:
        # print("original:", sentence)
        sentence_to_replace = sentence
        for phrase in phrases:
            if phrase.phrase.lower() == 'to':
                continue
            if len(phrase.meta_mappings) == 0:
                continue
            substitute_with = []
            should_substitute = False
            for mm in phrase.meta_mappings[0]:
                prefix = get_prefix(mm.group, substitutions_list)
                if prefix:
                    should_substitute = True
                    substitute_with.append(prefix.replace(',', '').replace(' ', ''))
                else:
                    substitute_with.append(mm.text)
            if should_substitute:
                substitute_with = ' '.join(substitute_with)
                # print("\treplacing:", phrase.phrase, "WITH:", substitute_with)
                sentence_to_replace = sentence_to_replace.replace(phrase.phrase, ' ' + substitute_with + ' ')
        # print("preferred:\t", sentence_to_replace)
        sentence_to_replace = sentence_to_replace.strip()
        sentence_to_replace = sentence_to_replace if sentence_to_replace[-1] == '.' else sentence_to_replace + '.'
        text = text.replace(sentence.strip(), ' ' + sentence_to_replace + ' ')

    print("preferred:\t", text.strip())
    # exit(1)
    return text.strip()


def disease_or_syndrome_replaced(op: Output, text: str) -> str:
    disease_or_syndrome = ['pathological function', 'disease or syndrome', 'mental or behavioral dysfunction',
                           'cell or molecular dysfunction', 'virus', 'neoplastic process', 'anatomic abnormality',
                           'injury or poisoning', 'congenital abnormality', 'acquired abnormality']

    return generic_replaced(op, text, disease_or_syndrome)


def most_replaced(op: Output, text: str) -> str:
    from metamap.groups_from_titles import substitutions
    return generic_replaced(op, text, substitutions)


def title_disease_or_syndrome_replaced(reference: Reference) -> str:
    op = read_metamapped_title_output(reference)
    return disease_or_syndrome_replaced(op, reference.title)


def abstract_disease_or_syndrome_replaced(reference: Reference) -> str:
    op = read_metamapped_abstract_output(reference)
    return disease_or_syndrome_replaced(op, reference.abstract)


def conclusions_disease_or_syndrome_replaced(reference: Reference) -> str:
    op = read_metamapped_abstract_output(reference)
    return disease_or_syndrome_replaced(op, reference.conclusions)


def methods_disease_or_syndrome_replaced(reference: Reference) -> str:
    op = read_metamapped_abstract_output(reference)
    return disease_or_syndrome_replaced(op, reference.methods)


def title_most_replaced(reference: Reference) -> str:
    op = read_metamapped_title_output(reference)
    text = reference.title

    return most_replaced(op, text)


def abstract_most_replaced(reference: Reference) -> str:
    op = read_metamapped_abstract_output(reference)
    text = reference.abstract

    return most_replaced(op, text)


def conclusions_most_replaced(reference: Reference) -> str:
    op = read_metamapped_abstract_output(reference)
    text = reference.conclusions

    return most_replaced(op, text)


def methods_most_replaced(reference: Reference) -> str:
    op = read_metamapped_abstract_output(reference)
    text = reference.methods

    return most_replaced(op, text)


def main():
    funs = [
        abstract_disease_or_syndrome_replaced,
        abstract_most_replaced,
        conclusions_disease_or_syndrome_replaced,
        conclusions_most_replaced,
        methods_disease_or_syndrome_replaced,
        methods_most_replaced,
        title_disease_or_syndrome_replaced,
        title_most_replaced,
    ]
    data = get_all_original_data()
    references = data.get_all_references().items()
    count = 0
    for key, reference in references:
        print("progress", count, '/', len(references), ' : ', count / len(references))
        for fun in funs:
            filepath = os.path.join(metamapped_dir, fun.__name__, key + '.txt')
            print(filepath)
            save_data_with_ultimate_dir_creation(filepath, [fun(reference)])
        count += 1


if __name__ == "__main__":
    main()
