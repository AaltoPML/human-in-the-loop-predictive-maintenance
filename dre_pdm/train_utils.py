# -*- coding: utf-8 -*-
import collections

import sklearn
import sklearn.preprocessing
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble
import sklearn.linear_model

import xgboost
import numpy as np
import copy

from dre_pdm import utils


RANDOM_SEEDS = [23, 42, 99, 101, 272,
                2 * 23, 2 * 42, 2 * 99, 2 * 101, 2 * 272,
                3 * 23, 3 * 42, 3 * 99, 3 * 101, 3 * 272,
                4 * 23, 4 * 42, 4 * 99, 4 * 101, 4 * 272,
                5 * 23, 5 * 42, 5 * 99, 5 * 101, 5 * 272]


MODELS = {
    "XGBoost": xgboost.XGBClassifier,
    "Extremely Randomized Trees": sklearn.ensemble.ExtraTreesClassifier,
    "Logistic Regression": sklearn.linear_model.LogisticRegression,
}


class DRE_classifier:
    """
    Implements DRE classifier.
    """
    def __init__(self, model_class, alpha=0.7):
        self.model_class = model_class
        self.model = None
        self.alpha = alpha

    def init_model(self, alpha_dre=None, **kwargs):
        self.model = self.model_class(**kwargs)
        self.alpha = alpha_dre

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, feedback=None):
        if feedback is None:
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            probabilities = self.alpha * self.model.predict_proba(X)[:, 1] + (1 - self.alpha) * feedback
        return np.array([round(p) for p in probabilities])


def _split_and_scale(X, y, feedback_results, test_size, random_state):
    if feedback_results is not None:
        X_train, X_val, y_train, y_val, f_res_train, f_res_val = sklearn.model_selection.train_test_split(
            X, y, feedback_results, test_size=test_size, random_state=random_state)
    else:
        X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        f_res_train, f_res_val = None, None
    scaler = sklearn.preprocessing.StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val, y_train, y_val, f_res_train, f_res_val


def _init_model(model_class, X_train, y_train, random_state=42, alpha_dre=None, n_jobs=None):
    cnt = collections.Counter(y_train)
    if model_class == xgboost.XGBClassifier:
        clf = model_class(scale_pos_weight=cnt[0] / cnt[1], random_state=random_state, n_jobs=n_jobs)
    elif model_class == sklearn.ensemble.ExtraTreesClassifier:
        clf = model_class(n_estimators=100, class_weight={0: cnt[1], 1: cnt[0]},
                          random_state=random_state, n_jobs=n_jobs, max_depth=20)
    elif model_class == sklearn.linear_model.LogisticRegression:
        clf = model_class(class_weight={0: cnt[1], 1: cnt[0]}, random_state=random_state, n_jobs=n_jobs)
    elif model_class == DRE_classifier:
        assert alpha_dre is not None
        clf = model_class(xgboost.XGBClassifier, alpha=alpha_dre)
        clf.init_model(scale_pos_weight=cnt[0] / cnt[1], random_state=random_state, n_jobs=n_jobs, alpha_dre=alpha_dre)
    else:
        raise Exception("Unsupported model type")
    return clf


def get_unified_feedback(feedbacks, X):
    """
    Calculatees application of feedback rules from `feedbacks` with inverse distance weights.
    """
    results = []
    weights = []
    for f in feedbacks:
        results.append(f._decision_rule.predict_proba(X)[:, 1])
        weights.append(1. / (f.dist(X) + 1e-5))
    return np.array(results).T, np.array(weights).T


def train_model(model_class, X, y, random_state=42, n_jobs=None, alpha_dre=None):
    """
    Trains a model from model_class on dataset (X, y).
    """
    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    clf = _init_model(model_class, X, y, random_state=random_state, n_jobs=n_jobs, alpha=alpha_dre)
    clf.fit(X, y)
    print("F1_score on training data: {}".format(sklearn.metrics.f1_score(clf.predict(X), y)))
    return clf, scaler


def validate_model_classification(
        model_class, X, y, humans=None, feedback_results=None,
        test_size=0.33, random_state=42, n_jobs=None, alpha_dre=None, ratio_to_look=None, additional_ds=None):
    """
    Validates a model from model_class on (X, y). Optionally uses list of humans for decision rule elication evaluation.
    """

    additional_ds = additional_ds or []
    X_train, X_val, y_train, y_val, feedback_train, feedback_val = _split_and_scale(
        X, y, feedback_results, test_size=test_size, random_state=random_state)

    for i in range(len(additional_ds)):
        X_new, y_new = additional_ds[i][0], additional_ds[i][1]
        X_train = np.concatenate((X_train, X_new), axis=0)
        y_train = np.concatenate((y_train, y_new), axis=0)

    clf = _init_model(model_class, X_train, y_train, random_state=random_state, n_jobs=n_jobs, alpha_dre=alpha_dre)

    clf.fit(X_train, y_train)

    datasets = None
    if model_class == DRE_classifier:
        assert humans is not None
        for i in range(len(humans)):
            humans[i].fit(X_train, y_train)
        feedbacks, datasets = utils.get_humans_feedback(
            humans, X_val, y_val, ratio_to_look, clf.model)
        feedback_val, feedback_weights = get_unified_feedback(feedbacks, X_val)
        if len(feedback_val):
            feedback_val = np.average(feedback_val, axis=1, weights=feedback_weights)
            ypred = clf.predict(X_val, feedback_val)
        else:
            ypred = clf.predict(X_val)
    else:
        ypred = clf.predict(X_val)

    f1_score = sklearn.metrics.f1_score(ypred, y_val)
    precision = sklearn.metrics.precision_score(ypred, y_val)
    recall = sklearn.metrics.recall_score(ypred, y_val)

    return f1_score, precision, recall, clf, datasets


def validate_model_classification_stats(
        model_class, X, y, humans=None, test_size=0.33,
        limit=None, n_jobs=None, alpha_dre=None, ratio_to_look=None, additional_ds=None):
    additional_ds = additional_ds or []

    f1_scores, precisions, recalls, all_datasets = [], [], [], []
    model = None
    if limit is not None:
        random_seeds = RANDOM_SEEDS[:limit]
    else:
        random_seeds = RANDOM_SEEDS
    for i, rs in enumerate(random_seeds):
        utils.set_all_random_seeds(rs)
        cur_add_ds = additional_ds[i] if len(additional_ds) else None
        f1_score, precision, recall, model, datasets = validate_model_classification(
            model_class, X, y, copy.deepcopy(humans), test_size=test_size, random_state=rs,
            n_jobs=n_jobs, alpha_dre=alpha_dre, ratio_to_look=ratio_to_look,
            additional_ds=cur_add_ds)

        print("Iteration: {}".format(i))
        print("F1_score: ", f1_score)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print()

        f1_scores.append(f1_score)
        precisions.append(precision)
        recalls.append(recall)
        all_datasets.append(datasets)

    return f1_scores, precisions, recalls, model, all_datasets


def validate_all_models(X, y, test_size=0.33):
    """
    Performs evaluation for all model classes from MODELS dict. 
    """
    results = {}
    for model_name, model_class in MODELS.items():
        print("Evaluating {}:".format(model_name))
        f1_scores, precisions, recalls, model = validate_model_classification_stats(model_class, X, y, test_size)
        print("F1 score: {} ± {}".format(round(np.mean(f1_scores), 4), round(np.std(f1_scores), 4)))

        print("Precision: {} ± {}".format(round(np.mean(precisions), 4),
                                          round(np.std(precisions), 4)))

        print("Recall: {} ± {}".format(round(np.mean(recalls), 4),
                                       round(np.std(recalls), 4)))
        print("===================")
        results[model_name] = {}
        results[model_name]["f1_score"] = f1_scores
        results[model_name]["precision"] = precisions
        results[model_name]["recall"] = recalls

    return results
