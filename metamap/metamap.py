# Call metamap free text indexer and process the results.

from typing import List, Optional, Tuple


def remove_non_ascii(s: str) -> str:
    return ''.join(filter(lambda x: ord(x) < 128 and x != '\'' and x != '\"', s))


def call_metamap(text: str) -> str:
    import subprocess
    meta_file = 'metamap/public_mm/bin/metamap18'
    text = text.strip()
    # metamap does not support non-ascii characters
    text = remove_non_ascii(text)

    cmd = 'echo "' + text + '" | ' + meta_file + ' -I'
    try:
        return subprocess.check_output("(" + cmd + ")", shell=True, stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        import sys
        print(text)
        print(cmd)
        sys.stderr.write(
            "common::run_command() : [ERROR]: output = %s, error code = %s\n"
            % (e.output, e.returncode))
        raise e


class MetaMapping:
    def __init__(self, text: str, preferred_name: Optional[str], group: str):
        self.text: str = text.lower()
        preferred_name = preferred_name if not preferred_name else preferred_name.lower()
        self.preferred_name: Optional[str] = preferred_name
        group = group if not group else group.lower()
        self.group: Optional[str] = group


class Phrase:
    def __init__(self, phrase: str):
        self.phrase: str = phrase
        self.meta_mappings: List[List[MetaMapping]] = []

    def add_meta_mappings(self, meta_mappings: List[MetaMapping]):
        self.meta_mappings.append(meta_mappings)


class Output:
    def __init__(self):
        self.phrases: List[Phrase] = []
        self.sentences: List[Tuple[str, List[Phrase]]] = []


def filter_out_empty_strings(a: List[str]) -> List[str]:
    return list(filter(lambda x: x != '', a))


def process_metamaping(meta_mapping):
    elems = meta_mapping.split('\n')[1:]
    elems = filter_out_empty_strings(elems)
    elems = list(map(lambda el: el.strip(), elems))
    meta_mappings_parsed = []
    for el in elems:
        el = el.split(':')[1]
        name = el
        preferred_name = None
        group = None
        if ' [' in el:
            parts = el.split(' [')
            parts = filter_out_empty_strings(parts)
            name = parts[0]
            el = parts[0]
            group = parts[1][:-1]

        if ' (' in el:
            parts = el.split(' (')
            name = parts[0]
            preferred_name = ' ('.join(parts[1:])[:-1]

        mm = MetaMapping(name, preferred_name, group)
        meta_mappings_parsed.append(mm)
    return meta_mappings_parsed


def split_sentence(op: Output, text: str) -> None:
    keyword = '\nPhrase: '
    phrases_texts = text.split(keyword)[1:]
    sentence = ': '.join(text.split(keyword)[0].split(': ')[1:])
    op.sentences.append((sentence, []))
    phrases_texts = filter_out_empty_strings(phrases_texts)
    for phrase_texts in phrases_texts:
        elems = phrase_texts.split('Meta Mapping ')
        phrase_text = elems[0].strip()
        meta_mappings = elems[1:]
        phrase = Phrase(phrase_text)
        for meta_mapping in meta_mappings:
            meta_mapping_parsed = process_metamaping(meta_mapping)
            phrase.add_meta_mappings(meta_mapping_parsed)
        op.phrases.append(phrase)
        op.sentences[-1][1].append(phrase)


def get_metamap_output(text: str) -> Output:
    op = Output()
    sentence_texts = text.split('Processing USER.tx.')
    sentence_texts = filter_out_empty_strings(sentence_texts)
    for sentence_text in sentence_texts:
        split_sentence(op, sentence_text)

    return op
