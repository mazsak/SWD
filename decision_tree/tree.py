import json
import math
import operator
from collections import Counter

import pandas as pd


class Node():

    def __init__(self, df, class_name, attribute=None, attribute_val=None, attribute_amount=None):
        self.df = df
        self.class_name = class_name
        self.is_leaf = False
        self.attribute = attribute
        self.attribute_val = attribute_val
        self.attribute_amount = attribute_amount
        self.attributes: pd.DataFrame = df[
            [c for c in df.columns.tolist() if c != class_name]]
        self.classes: pd.Series = df[class_name]
        self.children = {}

    def calculate(self):
        if len(set(self.classes.to_list())) == 1:
            self.is_leaf = True
            return
        else:
            counter = Counter(self.classes.to_list())
            i_value = -sum([math.log(v / len(self.classes), 2) * (v / len(self.classes)) for v in counter.values()])
            g_values = {}
            for att in self.attributes.columns.tolist():
                if self.attributes.dtypes[att] == 'float':
                    continue
                p_values = Counter(self.attributes[att])
                p_values = {k: v / len(self.df) for k, v in p_values.items()}
                classes = {}
                for k, v in p_values.items():
                    classes[k] = self.df[self.df[att] == k]
                e_values = {}
                for k, v in classes.items():
                    new_counter = Counter(v[self.class_name].to_list())
                    e_values[k] = -sum(
                        [math.log(vals / sum(new_counter.values()), 2) * (vals / sum(new_counter.values())) for vals in
                         new_counter.values()])
                final_e = sum([p_values[k] * e_values[k] for k in p_values.keys()])
                g_values[att] = i_value - final_e
            if all(g_val == 0 for g_val in g_values.values()):
                self.is_leaf = True
                return
            chosen_attribute = max(g_values.items(), key=operator.itemgetter(1))
            self.attribute = chosen_attribute[0]
            attribute_keys = list(Counter(self.attributes[self.attribute]).keys())
            for key in attribute_keys:
                self.children[key] = Node(self.df[self.df[self.attribute] == key], self.class_name, self.attribute, key)
                self.children[key].attribute_amount = Counter(self.children[key].df[self.class_name].to_list())
            for child in self.children.values():
                child.calculate()

    def to_string(self, i) -> None:
        print("\t" * i + f"{i}: CLASS_NAME:{self.class_name}, is_leaf:{self.is_leaf}, attribute:{self.attribute},"
                         f" attribute_amount:{self.attribute_amount}, attribute_val={self.attribute_val}")
        for x in self.children.values():
            x.to_string(i + 1)

    def to_txt(self, i, file) -> None:
        file.write("\t" * i + f"{i}: CLASS_NAME:{self.class_name}, is_leaf:{self.is_leaf}, attribute:{self.attribute},"
                              f" attribute_amount:{self.attribute_amount}, attribute_val={self.attribute_val}\n")
        for x in self.children.values():
            x.to_txt(i + 1, file)

    def classify(self, row):
        if self.is_leaf:
            return self.attribute_amount.most_common(1)[0][0]
        if self.attributes.dtypes[self.attribute].name == 'category' and '_binned_' in self.attribute:
            real_name = self.attribute.split('_')[0]
            for child, node in self.children.items():
                if row[real_name] in child:
                    return node.classify(row)
        else:
            for child, node in self.children.items():
                if row[self.attribute] in child:
                    return node.classify(row)


class Tree:

    def __init__(self, df: pd.DataFrame, class_name):
        self.df = df
        self.class_name = class_name
        self.root = None

    def node_to_json(self, node):
        dictionary = {"attribute_name": str(node.attribute),
                      "attribute_amount": node.attribute_amount,
                      "attribute_val": str(node.attribute_val),
                      "class_name": node.class_name,
                      "is_leaf": node.is_leaf,
                      "children": [self.node_to_json(child) for child in node.children.values()]}
        return dictionary

    def calculate(self):
        self.root = Node(self.df, self.class_name)
        self.root.calculate()
        json_repr = self.node_to_json(self.root)
        with open("tree.json", 'w') as file:
            json.dump(json_repr, file)

    def to_string(self):
        self.root.to_string(0)

    def classify(self, row):
        prediction = self.root.classify(row)
        return prediction, row[self.class_name], row[self.class_name] in prediction if prediction else False

    def to_txt(self):
        with open('tree.txt', 'w') as f:
            self.root.to_txt(0, f)
