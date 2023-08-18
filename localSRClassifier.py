import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
import itertools
import time
import warnings
from itertools import product
from tqdm import tqdm

warnings.filterwarnings("ignore")

operators = {"Pow": {"arity":2, "symmetric": False}, "Mul": {"arity":2, "symmetric": True},
             "Add": {"arity":2, "symmetric": True}, "Sub": {"arity":2, "symmetric": False},
             "Div": {"arity":2, "symmetric": False},}

def Pow(op1, op2):
    if op1>0:
        return np.power(op1, op2)
    else:
        return -np.power(-op1, op2)
Pow = np.vectorize(Pow)

def Mul(op1, op2):
    return op1 * op2

def Add(op1, op2):
    return op1 + op2

def Sub(op1, op2):
    return op1 - op2

def Div(op1, op2):
    return op1 / op2

class Node:
    def __init__(self, symbol="UNFILLED", parent=None):
        self.symbol = symbol
        self.parent = parent
        self.children = []
        self.on_variable_path = False

def kexp_to_tree(kexp, ordered_symbols = True):
    kexp = list(kexp)
    root = Node()
    queue = [root]  # FIFO queue
    seen = [root]  # Tracker
    for symbol in kexp:
        if not queue:
            break
        cur_Node = queue[0]
        queue = queue[1:]
        cur_Node.symbol = symbol
        if symbol in operators:
            no_of_children = operators[symbol]["arity"]
            cur_Node.children = [Node(parent=cur_Node) for i in range(no_of_children)]
            queue.extend(cur_Node.children)
            seen.extend(cur_Node.children)
        elif symbol == "R":
            pass
    if ordered_symbols:
        queue = [root]  # FIFO queue
        all_nodes = [root]
        while queue:
            cur_Node = queue[0]
            queue = queue[1:]
            queue.extend(cur_Node.children)
            all_nodes.extend(cur_Node.children)
        for node in all_nodes:
            if node.symbol!="R" and operators[node.symbol]["symmetric"]:
                node.children = sorted(node.children, key=lambda x:x.symbol)
    return root

def tree_to_exp(node):
    symbol = node.symbol
    if node.symbol in operators:
        return (
            symbol
            + "("
            + "".join([tree_to_exp(child) + "," for child in node.children])[:-1]
            + ")"
        )
    else:
        return symbol

def cost(x, xdata, ydata, lambda_string):  # simply use globally defined x and y
    y_pred = eval(lambda_string)(x, xdata)
    return np.mean(((y_pred - ydata)/ydata)**2) # quadratic cost function

def func(xdata, a, b, c):
    return np.power(xdata[0,:],a)+b/xdata[1,:]

def get_exp_set(k_exp_front_length = 3):
    exhuastive_symbol_set = [i for i in operators]+["R"]
    k_exp_list = [list(i)+["R"]*(k_exp_front_length+1) for i in product(exhuastive_symbol_set,repeat=k_exp_front_length)]
    exp_list = [tree_to_exp(kexp_to_tree(list(i)+["R"]*(k_exp_front_length+1))) for i in product(exhuastive_symbol_set,repeat=k_exp_front_length)]
    exp_set = set(exp_list)
    return exp_set

from tqdm import tqdm

exp_set = get_exp_set(2)

def cost(x, xdata, ydata, lambda_string):  # simply use globally defined x and y
    y_pred = eval(lambda_string)(x, xdata)
    return np.mean(((y_pred - ydata) / ydata) ** 2)  # quadratic cost function

def func(xdata, a, b, c):
    return np.power(xdata[0, :], a) + b / xdata[1, :]


def ex_search(
    xdata,
    ydata,
    string_type="BFGS",
    divider=3,
    old_method=False,
    cost=cost,
    cost_2=cost,
):
    def random_constant():
        return np.random.randint(1, 31) / 10 if np.random.randint(3) else 0

    master_list = []
    num_of_feature = xdata.shape[0]

    for i in range(1):
        mse_tuple = tuple()

        for test_eq in tqdm(exp_set):
            test_eq_orig = test_eq
            R_count = test_eq.count("R")

            lambda_string = "lambda x,xdata:"

            if old_method:
                possible_combi_of_num = np.array(
                    list(itertools.product(range(num_of_feature + 1), repeat=R_count))
                )
                if len(possible_combi_of_num) > 100:
                    possible_combi_of_num = possible_combi_of_num[
                        np.random.choice(len(possible_combi_of_num), 100, replace=False)
                    ]
                for combi_var in possible_combi_of_num:
                    lambda_string = "lambda x,xdata:"
                    test_eq = test_eq_orig
                    index = 0
                    for i in combi_var:
                        if i == num_of_feature:
                            test_eq = test_eq.replace("R", f"x[{index}]", 1)
                            index += 1
                        else:
                            test_eq = test_eq.replace("R", f"xdata[{i}]", 1)
                    lambda_string += test_eq
                    try:
                        res = minimize(
                            cost,
                            x0=[random_constant() for i in range(index + 1)],
                            args=(xdata, ydata, lambda_string),
                            method=string_type,
                        )  # used to increase step size since classification algo doesnt change
                        optimized_cost = cost_2(res.x, xdata, ydata, lambda_string)
                        master_list.append(
                            (
                                test_eq_orig,
                                lambda_string,
                                res.x,
                                res.nit,
                                optimized_cost,
                            )
                        )
                    except RuntimeError:
                        print("No fit found")
                    except:
                        #print("SKIPPED")
                        pass
    else:
        return min(master_list, key=lambda x: x[4])

"""Implementation of the SR-Enhanced CART algorithm."""
import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness
from copy import deepcopy
import pandas as pd


class Node:
    def __init__(self, predicted_class, class_stats):
        self.predicted_class = predicted_class
        self.class_stats = class_stats
        self.feature_eq = None
        self.threshold = None
        self.left = None
        self.right = None


class SRClassifier:
    def __init__(self, max_depth=None, random_state=0):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        self.n_classes_ = len(set(list(np.array(y).flatten())))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _best_split(self, X, y):
        if len(X) <= 0:
            return None, None

        def _gini_formula(y, y_pred, w, mode):
            """Calculate the gini."""
            m = y.size
            if m <= 1:
                return 1
            num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
            best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
            best_idx, best_thr = None, None
            thresholds, classes = zip(*sorted(zip(y_pred, y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
            if mode == "threshold":
                if not best_thr:
                    return 0
                return best_thr
            elif mode == "gini":
                return best_gini
            elif mode == "dist":
                if not best_thr:
                    return np.inf
                seperation_array = y_pred > best_thr
                left_node = np.array(y_pred)[seperation_array]
                right_node = np.array(y_pred)[~seperation_array]
                for indiv_class in range(self.n_classes_):
                    is_left_more = np.sum(left_node == indiv_class) / len(
                        left_node
                    ) > np.sum(right_node == indiv_class) / len(right_node)
                    if is_left_more:
                        best_dist = np.sum(
                            np.abs(
                                right_node[
                                    np.array(y)[~seperation_array] == indiv_class
                                ]
                                - best_thr
                            )
                            ** 2
                        )
                    else:
                        best_dist = np.sum(
                            np.abs(
                                left_node[np.array(y)[seperation_array] == indiv_class]
                                - best_thr
                            )
                            ** 2
                        )
                return best_dist

        def new_cost(x, xdata, ydata, lambda_string):
            y_pred = eval(lambda_string)(x, xdata)
            if np.isnan(y_pred).any():
                return np.inf
            return _gini_formula(ydata, y_pred, None, "dist")

        def new_cost_2(x, xdata, ydata, lambda_string):
            y_pred = eval(lambda_string)(x, xdata)
            if np.isnan(y_pred).any():
                return np.inf
            return _gini_formula(ydata, y_pred, None, "gini")

        best_eq = ex_search(
            X.T,
            y,
            cost=new_cost,
            cost_2=new_cost_2,
            string_type="BFGS",
            old_method=True,
        )  # change this

        def calc_cost(x, xdata, ydata, lambda_string):
            y_pred = eval(lambda_string)(x, xdata)
            threshold = _gini_formula(ydata, y_pred, None, "threshold")
            return threshold

        best_thr = calc_cost(
            best_eq[2], X.T, y, best_eq[1]
        )  # transpose because of how exSR is written

        return best_eq, best_thr

    def _grow_tree(self, X, y, depth=0):
        class_stats = [f"Class{i}: {np.sum(y == i)}" for i in range(self.n_classes_)]
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class, class_stats=class_stats)
        if depth < self.max_depth and max(num_samples_per_class) < len(y) * 0.95:
            best_eq, best_thr = self._best_split(X, y)
            if best_eq is not None and best_thr is not None:
                indices_left = (
                    eval(best_eq[1])(best_eq[2], X.T) < best_thr
                )  # transpose because of how our eq is defined
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_eq = best_eq
                node.threshold = best_thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if (
                eval(node.feature_eq[1])(node.feature_eq[2], np.array([inputs]).T)
                < node.threshold
            ):
                node = node.left
            else:
                node = node.right
        return node.predicted_class
