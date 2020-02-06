from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from utils.feature_names import *
from features.text_processor import TextTokenizer
from params import MultiBaseParams
from utils.balancing import get_balancing_step


def get_numeric_transformer(method):
    from imblearn.pipeline import Pipeline
    from features.text_processor import PublicationYearTransformer
    steps = [('t', PublicationYearTransformer())]
    if method != "none":
        steps.append(("os", get_balancing_step(method)))

    pipeline = Pipeline(steps)
    return pipeline


def get_text_transformer(method: str, pca: bool, ngram_range):
    from imblearn.pipeline import Pipeline

    from sklearn.decomposition import TruncatedSVD
    steps = [
        ('vect', CountVectorizer(tokenizer=TextTokenizer().preprocess)),
        ('tfidf', TfidfTransformer())]

    if method != "none":
        steps.append(("os", get_balancing_step(method)))

    if pca:
        steps.append(('pca', TruncatedSVD()))

    pipeline = Pipeline(steps)
    pipeline.set_params(vect__ngram_range=ngram_range, tfidf__use_idf=True)
    return pipeline


def get_features_pipeline(params: MultiBaseParams):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    numeric_features = [publication_year_key]
    numeric_transformer = get_numeric_transformer(params.method)

    text_features_1_4 = [
        abstract_key,
        abstract_disease_or_syndrome_replaced_key,
        abstract_most_replaced_key,

        conclusions_key,
        conclusions_disease_or_syndrome_replaced_key,
        conclusions_most_replaced_key,

        journal_title_key,

        methods_key,
        methods_disease_or_syndrome_replaced_key,
        methods_most_replaced_key,

        title_key,
        title_disease_or_syndrome_replaced_key,
        title_most_replaced_key]
    text_transformer_1_4 = get_text_transformer(params.method, params.pca, ngram_range=(1, 4))
    text_features_1_1 = [
        mesh_headings_key,
        publication_type_key,
    ]
    text_transformer_1_1 = get_text_transformer(params.method, params.pca, ngram_range=(1, 1))

    transformers = []

    for feature_name in params.features:
        if feature_name in numeric_features:
            transformers.append((feature_name, numeric_transformer, feature_name))
        elif feature_name in text_features_1_1:
            transformers.append((feature_name, text_transformer_1_1, feature_name))
        elif feature_name in text_features_1_4:
            transformers.append((feature_name, text_transformer_1_4, feature_name))

    preprocessor = ColumnTransformer(transformers=transformers)

    steps = [('preprocessor', preprocessor)]
    # TODO balancing step here, not in column transformers
    from classifier.models import get
    parameters = {}
    if params.classifier.startswith('SVC'):
        parameters['probability'] = True
    if params.get_balanced():
        parameters['class_weight'] = 'balanced'
    steps.append(('clf', get(params.classifier, params=parameters)))

    pipeline = Pipeline(steps=steps)

    return pipeline
