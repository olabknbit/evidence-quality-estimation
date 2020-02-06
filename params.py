from abc import abstractmethod
from typing import List, Dict, Any


class Params:
    def __init__(self, features: List[Any], classifier: str, dirname: str, params: Dict[str, Any], pca: bool,
                 stacked: bool, use_only_ab: bool, to_binary: bool):
        self.features: List[Any] = features
        self.classifier = classifier
        self.dirname = dirname
        self.method = "undersampling" if "undersampling" in dirname else \
            ("oversampling" if "oversampling" in dirname else
             ("combine" if "combine" in dirname else "none"))
        self.params = params
        self.pca = pca
        self.stacked = stacked
        self.use_only_ab = use_only_ab
        self.to_binary = to_binary

    @abstractmethod
    def get_features_names_list(self) -> List[str]:
        pass

    def get_balanced(self) -> bool:
        return self.params['balanced'] if 'balanced' in self.params else False


class BaseParams(Params):
    def __init__(self, features: List[str], classifier: str, dirname: str, params: Dict[str, Any], pca: bool):
        super().__init__(features=features, classifier=classifier, dirname=dirname, params=params, pca=pca,
                         stacked=False,
                         use_only_ab=False, to_binary=False)

    def get_features_names_list(self) -> List[str]:
        return self.features

    def get_features_names_str(self) -> str:
        return '__'.join(self.features)

    def serialize(self):
        import pickle
        import os.path
        from ebm.filenames import get_features_dir
        path = os.path.join(get_features_dir(self.get_features_names_str(), self.dirname, "test", self.classifier),
                            'params')
        with open(path, 'wb') as config_dictionary_file:
            pickle.dump(self, config_dictionary_file)


def deserialize(feature_name_str, dirname, classifier) -> BaseParams:
    import pickle
    import os.path
    from ebm.filenames import get_features_dir
    path = os.path.join(get_features_dir(feature_name_str, dirname, "test", classifier), 'params')
    with open(path, 'rb') as config_dictionary_file:
        return pickle.load(config_dictionary_file)


class SingleBaseParams(BaseParams):
    def __init__(self, feature_name: str, classifier: str, dirname: str, params: Dict[str, Any], pca: bool):
        super().__init__(features=[feature_name], classifier=classifier, dirname=dirname, params=params, pca=pca)

    def get_feature_name(self):
        return self.features[0]


class MultiBaseParams(BaseParams):
    def __init__(self, features: List[str], classifier: str, dirname: str, params: Dict[str, Any], pca: bool):
        super().__init__(features=features, classifier=classifier, dirname=dirname, params=params, pca=pca)


class StackedParams(Params):
    def __init__(self, features: List[BaseParams], classifier: str, dirname: str, params: Dict[str, Any], pca: bool,
                 use_only_ab: bool, to_binary: bool):
        super().__init__(features=features, classifier=classifier, dirname=dirname, params=params, pca=pca,
                         stacked=True, use_only_ab=use_only_ab, to_binary=to_binary)

    def get_features_names_list(self):
        return [fn.get_features_names_str() for fn in self.features]
