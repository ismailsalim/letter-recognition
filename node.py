class Node:
    """
    A node of the decision tree that contains the relevant decision logic.

    Attributes:
    ----------------------------------
    feature_index_split : int
        The column index (zero-indexed) of the feature matrix that is used for
        the node's data binary splitting decision.

    integer_splitting_rule : int
        The integer that is used for the binary splitting decision.

    left_child : Node
        The left child of the node.

    right_child : Node
        The right child of the node.

    entropy: Int
        The entropy of given node's associated data.

    is_leaf : bool
        True if the following conditions hold when creating the node:
            1. All samples have the same class label.
            2. Data can't be split any further.
        False otherwise.

    predicted_class : char
        The most frequent character class in the data associated with a leaf
        node.
    """

    def __init__(self, features, labels, depth):
        self.features = features
        self.labels = labels
        self.feature_index_split = None
        self.integer_splitting_rule = None
        self.left_child = None
        self.right_child = None
        self.entropy = 0
        self.predicted_class = None
        self.is_leaf = False
        self.depth = depth


    # def print_node(self):
    #     if self.is_leaf:
    #         print("Predicted class: ", self.predicted_class)
    #     else:
    #         print("Feature index split: ", self.feature_index_split,
    #               " ; Integer splitting rule:", self.integer_splitting_rule)

