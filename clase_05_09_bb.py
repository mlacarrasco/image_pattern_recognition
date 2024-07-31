# Author: Manohar Mukku
# Date: 15.09.2018
# Desc: Branch and Bound implementation for feature selection
# Github: https://github.com/manoharmukku/branch-and-bound-feature-selection


import random
import queue
import itertools
import numpy as np

def criterion_function(features):
    return sum(features)

    '''
    # Squared criterion function
    result = 0
    for feat in features:
        result += feat**2
    return result
    '''
    
flag = True
J_max = -1
result_node = None

def isMonotonic(features):
    features = sorted(features)

    # Generate the powerset of the features
    powerset = []
    for i in range(1, len(features)+1):
        subset = itertools.combinations(features, i)
        powerset.extend(subset)

    # For all possible subset pairs, check if monotonicity is satisfied
    # print (powerset)
    for i, item1 in enumerate(powerset):
        for item2 in powerset[i+1:]:
            if (set(item1).issubset(set(item2)) and (criterion_function(list(item1)) > criterion_function(list(item2)))):
                return False

    return True

class tree_node(object):
    def __init__(self, value, features, preserved_features, level):
        self.branch_value = value
        self.features = features
        self.preserved_features = preserved_features
        self.level = level
        self.index = None
        self.children = []
        self.J = None


def branch_and_bound(root, D, d):
    global flag
    global J_max
    global result_node

    # Compute the criterion function
    root.J = criterion_function(root.features)

    # Stop building children for this node, if J <= J_max
    if (flag == False and root.J <= J_max):
        print('.', end='')
        return

    # If this is the leaf node, update J_max, result_node and return
    if (root.level == D-d):
        if (flag == True):
            J_max = root.J
            flag = False
            result_node = root

        elif (root.J > J_max):
            J_max = root.J
            result_node = root

        return

    # Compute the number of branches for this node
    no_of_branches = (d + 1) - len(root.preserved_features)

    # Generate the branches
    branch_feature_values = sorted(random.sample(list(set(root.features)-set(root.preserved_features)), no_of_branches))

    # Iterate on the branches, and for each branch, call branch_and_bound recursively
    for i, branch_value in enumerate(branch_feature_values):
        child = tree_node(branch_value, [value for value in root.features if value != branch_value], \
            root.preserved_features + branch_feature_values[i+1:], root.level+1)

        root.children.append(child)

        branch_and_bound(child, D, d)

def give_indexes(root):
    bfs = queue.Queue(maxsize=40)

    bfs.put(root)
    index = -1
    while (bfs.empty() == False):
        node = bfs.get()
        node.index = index
        index += 1
        for child in node.children:
            bfs.put(child)

def display_tree(node, dot_object, parent_index):
    # Create node in dob_object, for this node
    dot_object.node(str(node.index), "Features = " + str(node.features) + "\nJ(Features) = " + str(node.J) + "\nPreserved = " + str(node.preserved_features))

    # If this is not the root node, create an edge to its parent
    if (node.index != -1):
        dot_object.edge(str(parent_index), str(node.index), label=str(node.branch_value))

    # Base case, when the node has no children, return
    if (len(node.children) == 0):
        return

    # Recursively call display_tree for all the childern of this node
    for child in reversed(node.children):
        display_tree(child, dot_object, node.index)


def parse_features(features_string):
    return sorted([float(str) for str in features_string.split(',')])


   
# Defaults
#features = "1,2,3,4,"
#features = parse_features(features)
features = list(np.arange(1,30))
D = len(features)
d = 4

# Create the root tree node
root = tree_node(-1, features, [], 0)

# Call branch and bound on the root node, and recursively construct the tree
branch_and_bound(root, D, d)

# Give indexes(numbers) for nodes of constructed tree in BFS order (used for Graphviz)
#give_indexes(root)

print ("Output")
print ("------")
print ("Features considered = {}".format(result_node.features))
print ("Criterion function value = {}".format(result_node.J))

