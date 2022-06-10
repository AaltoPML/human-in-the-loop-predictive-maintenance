import copy
import random
import numpy as np

import matplotlib.pyplot as plt

from dre_pdm import feedback


def generate(mean, cov, n):
    """
    Generates n samples from multivariate normal distributions.
    """
    return np.random.multivariate_normal(mean, cov, n)


def set_all_random_seeds(random_seed):
    """
    Set random seeds in all the used libraries.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)


def visualize_tree_decision_boundary(clf, xmin, xmax, ymin, ymax, plot_step=1):
    """
    Visaulizes decision tree boundaries.
    """
    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, plot_step), np.arange(ymin, ymax, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    if not callable(clf):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = np.array([clf(point) for point in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)
    _ = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)


def subsample(X, labels, ratio):
    """
    Subsamples form X and labels with the given ratio.
    """
    N = len(X)
    idx = np.random.choice(np.arange(N), int(N*ratio), replace=False)
    return X[idx], labels[idx]


def test_model(X, labels, model):
    """
    Tests the model and returns all samples where the classifier was wrong with its labels.
    """
    incorrect_ids = np.where(model.predict(X) != labels)
    print('Accuracy of the model:', (len(X) - len(incorrect_ids)) / len(X))
    return X[incorrect_ids], labels[incorrect_ids]


def get_humans_feedback(humans, ds_for_tests, labels_for_tests, ratio_to_look, model):
    """
    Retrieves human feedback from `humans`.
    Uses a `ratio_to_look` fraction of `ds_for_tests` and `labels_for_tests` for checking the model's predictions.
    """
    feedbacks = set()

    datasets = []
    for j, human in enumerate(humans):
        print("Human #{}".format(j))

        X_cur, labels_cur = subsample(ds_for_tests, labels_for_tests, ratio_to_look)
        wrong_examples, we_true_labels = test_model(
            X_cur, labels_cur, model)
        if len(wrong_examples):
            predictions = (human.predict(wrong_examples) == we_true_labels)
        else:
            predictions = []
        cnt_surp, cnt_non_surp = 0, 0
        for i, p in enumerate(predictions):
            if p:
                cnt_non_surp += 1
                feedbacks.add(feedback.Feedback(human._tree, wrong_examples[i]))
            else:
                cnt_surp += 1
                human_tmp = copy.deepcopy(human)
                human_tmp.refit(
                    np.array([wrong_examples[i]]), np.array([we_true_labels[i]]))
                if human_tmp.predict([wrong_examples[i]]) == we_true_labels[i]:
                    feedbacks.add(feedback.Feedback(human_tmp._tree, wrong_examples[i]))
                humans[j] = human_tmp

        datasets.append((wrong_examples, we_true_labels))
        print("Total surprising data points: ", cnt_surp)
        print("Total non surprising examples: ", cnt_non_surp)
    return feedbacks, datasets
