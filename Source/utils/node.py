import numpy as np


class Node:

    def __init__(self, attribute=None, avail_attrs=[], depth=None, children=dict(), case_ids=[], is_leaf=False):
        self.is_leaf = is_leaf
        self.case_ids = case_ids
        self.attribute = attribute
        self.avail_attrs = avail_attrs
        self.children = children

        self.depth = depth

    def add_child(self, value, child):
        self.children[value] = child

    def add_case(self, case):
        self.case_ids.append(case)

    def set_cases(self, case_ids):
        self.case_ids = case_ids

    def set_terminal(self, case_ids):
        self.is_leaf = True
        self.case_ids = case_ids

    def get_instances(self):
        out = set()
        for val in self.children.values():
            if isinstance(val, Node):
                out.update(val.get_instances())
            else:
                out.update(set(val))

        return list(out)

    def retrieve_best(self, new_case, models, data, attr_types):
        object = self
        feat = object.attribute
        instances_ant = []
        while (object.is_leaf != True) and (len(object.case_ids) > 0):
            distances, closecat = self.compute_dist_v2(new_case[feat], models[feat], object.children, feat, attr_types)
            if attr_types[feat] == 'num_continuous':
                featvals = np.argsort(distances[:, 0])
                object = object.children[featvals[0]]
            elif attr_types[feat] == 'categorical':
                instances_ant = object.case_ids
                object = object.children[closecat]
            feat = object.attribute
        if len(object.case_ids) > 0:
            return data[object.case_ids,:]
        else:
            return data[instances_ant,:]

    def compute_dist(self, inst1, inst2, categories, feat, attr_types):
        distances = []
        closecat = ''
        if attr_types[feat] == 'num_continuous':
            for i in range(inst2.cluster_centers_.shape[0]):
                distances.append(np.abs(inst1 - inst2.cluster_centers_[i,0]))
        elif attr_types[feat] == 'categorical':
            closecat = inst1
            categ = list(categories.keys())
            if feat == 2:
                intersectt = []
                genres_newcase = inst1.split(',')
                for i in range(len(categ)):
                    genres_possible = categ[i].split(',')
                    intersectt.append(len(set(genres_newcase).intersection(set(genres_possible))))
                sort_index = np.argsort(np.array(intersectt))
                closecat = categ[sort_index[-1]]

        return np.array(distances).reshape((len(distances),1)), closecat

    def compute_dist_v2(self, inst1, inst2, categories, feat, attr_types):
        distances = []
        closecat = ''
        if attr_types[feat] == 'num_continuous':
            for i in range(inst2.cluster_centers_.shape[0]):
                distances.append(np.abs(inst1 - inst2.cluster_centers_[i,0]))
        elif attr_types[feat] == 'categorical':
            closecat = inst1
        return np.array(distances).reshape((len(distances),1)), closecat
