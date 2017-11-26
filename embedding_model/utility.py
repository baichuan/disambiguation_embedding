import math
import numpy as np


def sigmoid(x):
    return float(1) / (1 + math.exp(-x))


def construct_doc_matrix(dict, paper_list):
    """
    construct the learned embedding for document clustering
    dict: {paper_index, numpy_array}
    """
    D_matrix = dict[paper_list[0]]
    for idx in xrange(1, len(paper_list)):
        D_matrix = np.vstack((D_matrix, dict[paper_list[idx]]))
    return D_matrix


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
