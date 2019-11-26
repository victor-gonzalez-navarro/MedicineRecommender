import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from Source.utils.node import Node
from Source.utils.preprocess import Preprocess
from sklearn.preprocessing import MinMaxScaler
from scipy.linalg import norm
from statistics import mode


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class CaseBase:

    def __init__(self, x, num_class):
        # Preprocess of the data to be stored in the Case Base
        self.num_class = num_class
        self.prep = Preprocess()
        self.attr_names, self.attr_vals, self.attr_types, self.sol_cols, columns_names = self.prep.extract_attr_info(x,
                                                                                                       self.num_class)

        self.scaler = MinMaxScaler()
        # self.scaler.fit(self.songs_info[self.sol_cols])
        self.x = x.values
        aux_x, self.attr_vals = self.prep.fit_predict(self.x[:, :-self.num_class], columns_names, n_clusters=5)

        self.tree = None
        self.feat_selected = np.zeros((self.x.shape[1], 1))  # Depth at which each feature is selected
        self.max_depth = aux_x.shape[1]                      # Maximum depth corresponds to the number of attributes
                                                             # (+ leaf)

        self.make_tree(self.x, aux_x)

    def retain(self, solution, new_case):
        pass

    def get_tree_depth(self, tree=None, depth=0):
        if tree is None:
            tree = self.tree

        if tree.is_leaf:
            return depth + 1

        depths = []
        for _, child_tree in tree.children.items():
            depths.append(self.get_tree_depth(child_tree, depth) + 1)

        return max(depths)

    def get_n_cases(self, tree=None):
        if tree is None:
            tree = self.tree

        if tree.is_leaf:
            return len(tree.case_ids)

        cases = 0
        for _, child_tree in tree.children.items():
            cases += self.get_n_cases(child_tree)

        return cases

    def make_tree(self, x, x_aux):
        self.tree = self.find_best_partition(x_aux, avail_attrs=list(range(x_aux.shape[1])), depth=0)
        self.tree.set_cases(list(range(x_aux.shape[0])))
        self.expand_tree(self.tree, x, x_aux, depth=1)

    def find_best_partition(self, x, avail_attrs, depth):
        best_score = -np.inf
        best_aux_score = -np.inf
        for feat_ix in avail_attrs:
            unique_vals, counts = np.unique(x[:, feat_ix], return_counts=True)
            score = len(unique_vals) / len(self.attr_vals[feat_ix]) - np.exp(len(self.attr_vals[feat_ix])-len(self.attr_vals[2]))

            # Check the number of possible values for this attribute are in the remaining dataset
            if score > best_score:
                best_feat_ix = feat_ix
                best_score = score
                best_aux_score = np.std(counts)
            elif score == best_score:
                # In case of draw, select the attribute which values cover the most similar number of instances
                aux_score = np.std(counts)
                if aux_score < best_aux_score:
                    best_feat_ix = feat_ix
                    best_score = score
                    best_aux_score = aux_score

        # Annotate the depth at which this feature has been selected
        self.feat_selected[best_feat_ix] = depth

        # Remove the attribute from the list of available attributes
        avail_attrs = [attr for attr in avail_attrs if attr != best_feat_ix]

        # Create the Node and add a child per value of the selected attribute
        out_node = Node(attribute=best_feat_ix, avail_attrs=avail_attrs, depth=depth, children={})
        for val in self.attr_vals[best_feat_ix]:
            out_node.add_child(val, np.argwhere(x[:, best_feat_ix] == val)[:, 0])

        return out_node

    def expand_tree(self, tree, x, x_aux, depth):
        for key, val in tree.children.items():
            prev_val = np.copy(val)
            if len(val) == 0:
                # If the split left this branch empty, set the terminal boolean to True without adding any case
                tree.children[key] = Node(is_leaf=True, depth=depth)
            elif depth == self.max_depth:
                # If the maximum depth has been reached, add the terminal cases in the leaf node
                terminal_cases = np.array(tree.case_ids)[prev_val].tolist()  # x[val, :].tolist()
                tree.children[key] = Node(case_ids=terminal_cases, is_leaf=True, depth=depth)
            else:
                # Otherwise, find the best partition for this leaf and expand the subtree
                tree.children[key] = self.find_best_partition(x_aux[val, :], tree.avail_attrs, depth)
                tree.children[key].set_cases(np.array(tree.case_ids)[prev_val].tolist())
                self.expand_tree(tree.children[key], x[val, :], x_aux[val, :], depth + 1)

        return

    def check_node(self, x_tst, tree):
        if tree.is_leaf:
            return tree.case_ids
        else:
            return self.check_node(x_tst, tree.children[x_tst[tree.attribute]])

    def print_tree(self):
        print()
        print('--------------------')
        print('--------------------')
        print('The Case Base is:')
        print('--------------------')
        print('--------------------')
        print()
        self.print_tree_aux('Root', self.tree)

    def print_tree_aux(self, branch, tree):
        if tree.is_leaf and tree.case_ids:
            first = True
            for case in tree.case_ids:
                if first:
                    print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + str(branch) +
                          '\033[0m\u291a\u27f6\tcase_\033[94m' + str(self.x[case, :]) + '\033[0m')
                    first = False
                else:
                    print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + ' ' * len(str(branch)) +
                          '\033[0m\u291a\u27f6\tcase_\033[94m' + str(self.x[case, :]) + '\033[0m')

        elif tree.is_leaf and not tree.case_ids:
            print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + str(branch) + '\033[0m\u291a\u27f6case_\033[94m' +
                  'No cases yet' + '\033[0m')

        else:
            print('\t\t\t\t|' * tree.depth + '\u2919\033[92m' + str(branch) + '\033[0m\u291a\u27f6attr_\033[1m' +
                  str(self.attr_names[tree.attribute]) + '\033[0m')
            for branch, child_tree in tree.children.items():
                self.print_tree_aux(branch, child_tree)

    # It follows the best path, and in case it arrives to a point with no more instance it returns the instances
    # included by the parent
    def retrieve(self, new_case):
        object = self.tree
        feat = object.attribute
        instances_ant = []
        while (object.is_leaf != True) and (len(object.case_ids) > 0):
            distances, closecat = self.compute_distances(new_case[feat], self.prep.models[feat], feat)
            instances_ant = object.case_ids
            if self.attr_types[feat] == 'num_continuous':
                featvals = np.argsort(distances[:, 0])
                object = object.children[featvals[0]]
            elif self.attr_types[feat] == 'categorical':
                object = object.children[closecat]
            feat = object.attribute
        if len(object.case_ids) > 0:
            return self.x[object.case_ids,:]
        else:
            return self.x[instances_ant,:]

    def update(self, retrieved_cases, new_case, num_attrib_solution):
        # solution = retrieved_cases[0, -num_attrib_solution:].reshape((1, num_attrib_solution))
        solution = [''] * num_attrib_solution
        for i in range(1,num_attrib_solution+1):
            solution[-i] = pd.Series(retrieved_cases[:, -i]).mode()[0]
        return solution


    def compute_distances(self, inst1, inst2, feat):
        distances = []
        closecat = ''
        if self.attr_types[feat] == 'num_continuous':
            for i in range(inst2.cluster_centers_.shape[0]):
                distances.append(np.abs(inst1 - inst2.cluster_centers_[i,0]))
        elif self.attr_types[feat] == 'categorical':
            closecat = inst1
        return np.array(distances).reshape((len(distances), 1)), closecat

    # PARTIAL MATCHING
    def retrieve_v2(self, new_case):
        retrieved_cases = np.empty((0, len(new_case)+self.num_class))
        object = self.tree
        feat = object.attribute
        instances_ant = []
        while (object.is_leaf != True) and (len(object.case_ids) > 0):
            distances, closecat, seclosecat = self.compute_distances_v2(new_case[feat], self.prep.models[feat],
                                object.children, feat)
            # Retrieve instances second best and then following the best path
            if self.attr_types[feat] == 'num_continuous':
                featvals = np.argsort(distances[:, 0])
                retr = object.children[featvals[1]].retrieve_best(new_case, self.prep.models, self.x, self.attr_types)
                retrieved_cases = np.append(retrieved_cases, retr, axis=0)
            elif self.attr_types[feat] == 'categorical' and seclosecat != '':
                retr = object.children[seclosecat].retrieve_best(new_case, self.prep.models, self.x, self.attr_types)
                retrieved_cases = np.append(retrieved_cases, retr, axis=0)
            instances_ant = object.case_ids
            if self.attr_types[feat] == 'num_continuous':
                object = object.children[featvals[0]]
            elif self.attr_types[feat] == 'categorical':
                object = object.children[closecat]
            feat = object.attribute
        if len(object.case_ids) > 0:
            return np.concatenate((self.x[object.case_ids,:], retrieved_cases), axis=0)
        else:
            return np.concatenate((self.x[instances_ant,:], retrieved_cases), axis=0)


    def compute_distances_v2(self, inst1, inst2, categories, feat):
        attr_not_compute_best = [0, 1, 2, 3, 9, 10, 21, 22]
        diction = dict()
        diction[4] = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)',
                      '[90-100)']
        diction[5] = np.sort(self.attr_vals[5])
        diction[6] = np.sort(self.attr_vals[6])
        diction[7] = np.sort(self.attr_vals[7])
        diction[17] = np.sort(self.attr_vals[17])
        diction[18] = np.sort(self.attr_vals[18])
        diction[19] = np.sort(self.attr_vals[19])

        distances = []
        seclosecat = ''
        closecat = ''
        if self.attr_types[feat] == 'num_continuous':
            for i in range(inst2.cluster_centers_.shape[0]):
                distances.append(np.abs(inst1 - inst2.cluster_centers_[i,0]))
        elif self.attr_types[feat] == 'categorical':
            closecat = inst1
            if feat not in attr_not_compute_best:
                idx = list(diction[feat]).index(inst1)
                if idx == len(diction[feat])-1:
                    seclosecat = diction[feat][idx-1]
                else:
                    seclosecat = diction[feat][idx+1]

        return np.array(distances).reshape((len(distances), 1)), closecat, seclosecat