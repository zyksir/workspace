import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier

# 使用 StratifiedKFold 保证了每次 Split 中各个类别的比例都是相似的
def get_out_of_fold_predictions(X, y, models):
    meta_X, meta_y = list(), list()
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    for train_ix, test_ix in kfold.split(X, y):
        fold_yhats = list()
        train_X, test_X, train_y, test_y = X[train_ix], X[test_ix], y[train_ix], y[test_ix]
        meta_y.extend(test_y)
        for model in models:
            model.fit(train_X, train_y)
            yhat = model.predict_proba(test_X)
            fold_yhats.append(yhat)
        meta_X.append(np.hstack(fold_yhats))
    return np.vstack(meta_X), np.asarray(meta_y)

def fit_base_models(X, y, models):
    for model in models:
        model.fit(X, y)

# 这里我们也可以用其他模型
def fit_meta_model(X, y):
    model = ExtraTreesClassifier()
    model.fit(X, y)
    return model

def make_predictions(X, models, meta_model):
    meta_X = list()
    for model in models:
        yhat = model.predict_proba(X)
        meta_X.append(yhat)
    meta_X = np.hstack(meta_X)
    return meta_model.predict(meta_X)

'''
models = get_models()
meta_X, meta_y = get_out_of_fold_predictions(train_X, train_y, models)
fit_base_model(train_X, train_y, models)
meta_model = fit_meta_model(meta_X, meta_y)
make_predictions(test_X, models, meta_model)
'''