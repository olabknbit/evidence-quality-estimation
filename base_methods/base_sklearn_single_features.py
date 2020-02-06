from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from ebm.dataset import Data
from features.get_from_reference import get_f
from features.text_processor import TextTokenizer
from params import SingleBaseParams
from utils.balancing import get_balancing_step
from utils.feature_names import *


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def get_feature_pipeline(params: SingleBaseParams):
    from imblearn.pipeline import Pipeline
    from sklearn.decomposition import TruncatedSVD

    feature_name = params.get_feature_name()

    if feature_name == publication_year_key:
        from features.text_processor import PublicationYearTransformer
        steps = [('t', PublicationYearTransformer())]
    else:
        steps = [
            ('vect', CountVectorizer(tokenizer=TextTokenizer().preprocess)),
            ('tfidf', TfidfTransformer()),
        ]
    if params.method != "none":
        steps.append(("os", get_balancing_step(params.method)))

    if params.pca:
        steps.append(('pca', TruncatedSVD()))

    from classifier.models import get
    parameters = {}
    if params.classifier.startswith('SVC'):
        parameters['probability'] = True
    if params.get_balanced():
        parameters['class_weight'] = 'balanced'
    steps.append(('clf', get(params.classifier, params=parameters)))

    pipeline = Pipeline(steps)
    if feature_name == publication_year_key:
        pass
    elif feature_name in [publication_type_key, mesh_headings_key]:
        pipeline.set_params(vect__ngram_range=(1, 1))
        pipeline.set_params(tfidf__use_idf=True)
    elif feature_name in [title_key, title_most_replaced_key, title_disease_or_syndrome_replaced_key,
                          journal_title_key]:
        pipeline.set_params(vect__ngram_range=(1, 2))
        pipeline.set_params(tfidf__use_idf=True)
    else:
        pipeline.set_params(vect__ngram_range=(1, 4))
        pipeline.set_params(tfidf__use_idf=True)

    return pipeline


def one_feature_pipeline_cross_val(train: Data, test: Data, params: SingleBaseParams) -> None:
    f = get_f(params.get_feature_name())
    X_train, y_train = train.get_x_y(f)
    X_test, y_test = test.get_x_y(f)

    pipeline = get_feature_pipeline(params)

    from utils.metrics import calculate_metrics
    from sklearn.model_selection import cross_val_score

    results = cross_val_score(pipeline, X_train, y_train, cv=10)
    print(params.method)
    print("Cross validated accuracy score: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    pipeline.fit(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    preds = pipeline.predict(X_test)

    m = calculate_metrics(y_test, preds)
    print(m.to_string())
    print("test score = %0.4f" % test_score)
    print(params.get_feature_name())

    with open("base_features_hyperparams/" + params.get_feature_name() + '.txt', 'a') as f:
        f.write("-------\n")
        f.write("score: %.2f%% (%.2f%%)\n" % (
            results.mean() * 100, results.std() * 100))
        f.write("test score = %0.4f\n" % test_score)
        f.write(m.to_string() + '\n')
        f.write("method %s\n" % params.method)
        f.write("pca" + str(params.pca) + '\n')
        f.write("-------\n")
