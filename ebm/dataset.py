import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

from ebm.filenames import clinical_inquiries_xml, abstracts_dir, alta_2011_shared_task_dataset_trainset, \
    alta_2011_shared_task_dataset_trainset_dir, alta_2011_shared_task_dataset_devtestset, \
    alta_2011_shared_task_dataset_devtestset_dir, alta_2011_shared_task_testset_dir, alta_2011_shared_task_testset
from ebm.reference import create_reference, Reference
from ebm.snippet import Snippet, create_snippet


class Data:
    def __init__(self):
        self.snippets: List[Snippet] = []

    def split_data_train_test(self) -> Tuple['Data', 'Data']:
        # In order to split data:
        # len(train) ~= 855
        # len(test) ~= 183
        # we split before record_id = 7546
        train, test = [], []
        train_done = False
        for snippet in self.snippets:
            if snippet.evidence_id == '7546':
                train_done = True
            if not train_done:
                train.append(snippet)
            else:
                test.append(snippet)
        train_data = Data()
        train_data.snippets = train

        test_data = Data()
        test_data.snippets = test
        return train_data, test_data

    def get_all_references(self) -> Dict[str, Reference]:
        references = {}
        for snippet in self.snippets:
            for reference in snippet.references:
                references[reference.id] = reference
        return references

    def split_data_train_dev_test(self) -> Tuple['Data', 'Data', 'Data']:
        # In order to split data:
        # len(train) ~= 677
        # len(dev) ~= 178
        # len(test) ~= 183
        # we split before record_id = 1816 and then before record_id = 7546
        train, dev, test = [], [], []
        train_done = False
        dev_done = False
        for i, snippet in enumerate(self.snippets):
            # TODO check if these are correct
            if snippet.evidence_id == '18161':
                train_done = True
            elif snippet.evidence_id == '75461':
                dev_done = True
            if not train_done:
                train.append(snippet)
            elif train_done and not dev_done:
                dev.append(snippet)
            else:
                test.append(snippet)
        train_data = Data()
        train_data.snippets = train

        dev_data = Data()
        dev_data.snippets = dev

        test_data = Data()
        test_data.snippets = test
        return train_data, dev_data, test_data

    def show_baseline(self) -> None:
        y_true = [s.sor_grade for s in self.snippets]
        y_pred = ['B' for _ in self.snippets]
        from utils.metrics import save_metrics
        save_metrics(y_pred, y_true, '', multiclass=True)

    def split_train_all_test_one(self, id: str) -> Tuple['Data', 'Data']:
        train_data = Data()
        train_data.snippets = list(filter(lambda x: x.get_evidence_id() != id, self.snippets))

        test_data = Data()
        test_data.snippets = list(filter(lambda x: x.get_evidence_id() == id, self.snippets))
        return train_data, test_data

    def split_batches(self, n, i):
        train_batch = Data()
        test_batch = Data()

        start_index = i * n

        end_index = (i + 1) * n
        test_batch.snippets = self.snippets[start_index: end_index]
        train_batch.snippets = self.snippets[:start_index] + self.snippets[end_index:]

        return train_batch, test_batch

    def get_X(self, f):
        return [' '.join(snippet.get_features(f)) for snippet in self.snippets]

    def get_y(self):
        Y = [snippet.sor_grade for snippet in self.snippets]
        return Y

    def get_x_y(self, f):
        X = self.get_X(f)
        Y = self.get_y()
        return X, Y

    def get_pandas_xy(self, fs):
        import pandas as pd
        Data = {f.__name__: self.get_X(f) for f in fs}

        X = pd.DataFrame(Data)
        y = pd.Series(self.get_y())
        return X, y

    def get_evidence_ids(self):
        evidence_ids = [s.evidence_id for s in self.snippets]
        return evidence_ids


def filter_out_references(reference_ids, abstract_directory):
    def is_reference(reference_id):
        from ebm.reference import create_reference
        reference = create_reference(reference_id, abstract_directory)
        return reference is not None

    return set(filter(is_reference, reference_ids))


def get_original_data(abstracts_dir: str, filename: str):
    data = Data()
    sor_texts = get_sor_texts()
    with open(filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            elems = line.strip().split(' ')
            evidence_id = elems[0]
            sor_grade = elems[1]
            reference_ids = elems[2:]
            references = []
            for ref_id in reference_ids:
                reference = create_reference(ref_id, abstracts_dir)
                if reference:
                    references.append(reference)
            if len(references) > 0:
                snippet = Snippet(evidence_id, sniptext='', sor_grade=sor_grade, references=references, question='',
                                  sor_text=sor_texts[evidence_id])
                data.snippets.append(snippet)

    return data


def get_train_data() -> Data:
    return get_original_data(alta_2011_shared_task_dataset_trainset_dir, alta_2011_shared_task_dataset_trainset)


def get_train_dev_data() -> Data:
    data = get_train_data()
    data.snippets.extend(get_dev_data().snippets)
    return data


def get_dev_data() -> Data:
    return get_original_data(alta_2011_shared_task_dataset_devtestset_dir, alta_2011_shared_task_dataset_devtestset)


def get_test_data() -> Data:
    return get_original_data(alta_2011_shared_task_testset_dir, alta_2011_shared_task_testset)


def get_all_original_data() -> Data:
    data = get_train_dev_data()
    data.snippets.extend(get_test_data().snippets)
    return data


# returns full Data object (with snippets and references fields)
@DeprecationWarning
def get_all_data() -> Data:
    data = Data()
    with open(clinical_inquiries_xml, 'r', encoding="utf-8") as f:
        root = ET.parse(f).getroot()

        for record in root:
            record_id = record.attrib['id']
            question = record.find('question').text
            answer = record.find('answer')
            snippets = answer.findall('snip')
            for snippet_xml in snippets:
                snippet = create_snippet(record_id, snippet_xml, question, abstracts_dir)

                if snippet is not None:
                    data.snippets.append(snippet)

    return data


def get_sor_texts() -> Dict[str, str]:
    sor_texts = {}
    with open(clinical_inquiries_xml, 'r', encoding="utf-8") as f:
        root = ET.parse(f).getroot()

        for record in root:
            record_id = record.attrib['id']
            answer = record.find('answer')
            snippets = answer.findall('snip')
            for snippet_xml in snippets:
                snip_id = snippet_xml.attrib['id']
                sor_text = snippet_xml.find('sor').text
                evidence_id = record_id + snip_id
                sor_texts[evidence_id] = sor_text

    return sor_texts
