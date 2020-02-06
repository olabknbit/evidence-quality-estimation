from utils.feature_names import *

SVC_pipeline_params = {
    abstract_key: {'svm__C': 1.0, 'tfidf__norm': 'l2', 'tfidf__use_idf': True, 'vect__max_df': 0.75,
                   'vect__ngram_range': (1, 4)},
    abstract_disease_or_syndrome_replaced_key: {'svm__C': 1.0, 'tfidf__norm': 'l2', 'tfidf__use_idf': False,
                                                'vect__max_df': 0.75, 'vect__ngram_range': (1, 4)},
    abstract_most_replaced_key: {'svm__C': 1.0, 'tfidf__norm': 'l2', 'tfidf__use_idf': False, 'vect__max_df': 0.75,
                                 'vect__ngram_range': (1, 4)},
    conclusions_key:
        {'svm__C': 1.0, 'tfidf__norm': 'l2', 'tfidf__use_idf': True, 'vect__max_df': 0.5, 'vect__ngram_range': (1, 1)},
    conclusions_disease_or_syndrome_replaced_key: {'tfidf__use_idf': False, 'vect__max_df': 0.75,
                                                   'vect__ngram_range': (1, 3)},
    conclusions_most_replaced_key: {'svm__C': 1.0, 'tfidf__norm': 'l2', 'tfidf__use_idf': False, 'vect__max_df': 0.75,
                                    'vect__ngram_range': (1, 1)},
    journal_title_key:
        {'svm__C': 1.0, 'tfidf__norm': 'l1', 'tfidf__use_idf': False, 'vect__max_df': 0.75,
         'vect__ngram_range': (1, 1)},
    mesh_headings_key: {'svm__C': 1.0, 'tfidf__norm': 'l1', 'tfidf__use_idf': True, 'vect__max_df': 0.5,
                        'vect__ngram_range': (1, 1)},
    methods_key: {'svm__C': 100.0, 'tfidf__norm': 'l1', 'tfidf__use_idf': False, 'vect__max_df': 0.75,
                  'vect__ngram_range': (1, 3)},
    methods_disease_or_syndrome_replaced_key: {'tfidf__use_idf': False, 'vect__max_df': 0.75,
                                               'vect__ngram_range': (1, 3)},
    methods_most_replaced_key: {'tfidf__use_idf': False, 'vect__max_df': 0.75,
                                'vect__ngram_range': (1, 3)},
    # "publication_year": {},
    publication_type_key: {'vect__ngram_range': (1, 4)},
    title_key: {'svm__C': 1.0, 'tfidf__norm': 'l2', 'tfidf__use_idf': False, 'vect__max_df': 0.5,
                'vect__ngram_range': (1, 4)},
    title_disease_or_syndrome_replaced_key:
        {'svm__C': 1.0, 'tfidf__norm': 'l2', 'tfidf__use_idf': False, 'vect__max_df': 0.5, 'vect__ngram_range': (1, 1)},
    title_most_replaced_key:
        {'svm__C': 10.0, 'tfidf__norm': 'l1', 'tfidf__use_idf': True, 'vect__max_df': 0.5, 'vect__ngram_range': (1, 1)},
}
