def get_balancing_step(method: str):
    from imblearn.under_sampling import NearMiss
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN
    if method == "undersampling":
        return NearMiss()
    elif method == "oversampling":
        return SMOTE()
    elif method == "combine":
        return SMOTEENN()
    else:
        return None
