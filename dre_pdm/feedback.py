import numpy as onp
from sklearn.tree import export_text


class Feedback:
    """
    Stores feedback rule with additional information: point, distance function,
    and experience of the expert who created this rule.
    """
    def __init__(self, decision_rule, point, experience=None, dist=onp.linalg.norm):
        self._decision_rule = decision_rule
        self._point = point
        self.dist_f = dist
        self._human_experience = experience

    def __str__(self):
        str_repr = export_text(self._decision_rule, feature_names=("x1", "x2"))
        return "Decision rule: {}, Point: {}".format(str_repr, self._point)

    def dist(self, x):
        return self.dist_f(x - self._point, axis=1)
