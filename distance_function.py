import math
from collections import Counter
from typing import List

import numpy as np


def calculate_distance(k, metric, class_name, new_row, data):
    for column in data.select_dtypes(['object', 'string']).columns:
        if column != class_name:
            values: List[str] = list(set(data[column].values))
            encoded_values = {x: i for i, x in enumerate(values)}
            data[column] = data[column].map(encoded_values)

    if metric == 'Euclidean':
        distance = distance_euclidean(class_name, data,new_row)
    elif metric == 'Manhattan':
        distance = distance_manhattan(class_name, data, new_row)
    elif metric == 'Chebyshev':
        distance = distance_chebyshev(class_name, data, new_row)
    elif metric == 'Mahalanobis':
        distance = distance_mahalanobis(class_name, data, new_row)
    distance.sort(key=lambda x: x['distance'])
    closest = distance[0: min(k, len(distance))]
    counter = Counter([x['y'] for x in closest])
    y = counter.most_common(1)

    return y[0][0]


def distance_mahalanobis(class_name, data, new_row):
    columns = data.columns.tolist()
    columns.remove(class_name)
    cov = data[columns].cov().to_numpy()
    inv_covmat = np.linalg.inv(cov)
    distance = [{'distance': np.subtract(np.array(list(new_row.values())), np.array(
        [row[name_column] for name_column in data.columns.tolist() if name_column != class_name])).T.dot(
        inv_covmat).dot(np.subtract(np.array(list(new_row.values())), np.array(
        [row[name_column] for name_column in data.columns.tolist() if name_column != class_name]))),
                 'y': row[class_name]} for row in data.iloc]
    return distance


def distance_chebyshev(class_name, data, new_row):
    return [{'distance': max(
        [abs(row[column_name] - new_row[column_name]) for column_name in data.columns.tolist()
         if
         column_name != class_name]), 'y': row[class_name]} for
        row in data.iloc]


def distance_manhattan(class_name, data, new_row):
    return [{'distance':
                 sum([abs(row[column_name] - new_row[column_name]) for column_name in
                      data.columns.tolist()
                      if
                      column_name != class_name]), 'y': row[class_name]} for
            row in data.iloc]


def distance_euclidean(class_name, data, new_row):
    return [{'distance': math.sqrt(
        sum([math.pow(row[column_name] - new_row[column_name], 2) for column_name in data.columns.tolist()
             if
             column_name != class_name])), 'y': row[class_name]} for
        row in data.iloc]
