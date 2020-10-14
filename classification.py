import numpy as np
from node import Node
from labels import *
from eval import Evaluator


class DecisionTreeClassifier(object):
    """
    A binary decision tree classifier.

    Attributes
    ----------
    is_trained : bool
        Keeps track of whether the classifier has been trained

    root : Node
        The root of the decision tree structure (i.e. a representation of the
        starting decision logic).

    Methods
    -------
    train(X, y)
        Constructs a decision tree from data X and label y
    predict(X)
        Predicts the class label of samples X

    """

    def __init__(self):
        self.is_trained = False
        self.root = None
        self.maximum_depth = 0

    def train(self, x, y):
        """
        Constructs a decision tree classifier from data.
        :param x:  numpy.array
            An N by K numpy array (N is the number of instances, K is the
            number of attributes)
        :param y: numpy.array
            An N-dimensional numpy array
        :return: DecisionTreeClassifier
            A copy of the DecisionTreeClassifier instance
        """
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        root_depth = 0
        self.root = self.__build_tree(x, y, root_depth)

        # set a flag so that we know that the classifier has been trained
        self.is_trained = True

        return self

    def __build_tree(self, x, y, depth):
        """
        Grows the decision tree recursively.
        :param x: numpy.array
                An N by K numpy array (N is the number of instances, K is the
                number of attributes)
        :param y: numpy.array
                An N-dimensional numpy array
        :return: Node
        The root node that represents the first splitting decision of the tree.
        """

        node = Node(x, y, depth)
        # update max depth
        if depth > self.maximum_depth:
            self.maximum_depth = depth

        classes = np.unique(y)
        class_counts = np.unique(y, return_counts=True)[1]

        # accounting for data inconsistency (such as identical feature
        # distribution but different assigned class)
        predicted_class = classes[np.argmax(class_counts)]
        feature, split = self.__find_best_split(x, y)

        # only assign a predicted class to leaf nodes
        if feature is None or split is None:
            node.is_leaf = True
            node.predicted_class = predicted_class
            return node

        node.feature_index_split = feature
        node.integer_splitting_rule = split
        node.entropy = self.__entropy(y)

        row_indices_left_child = x[:, feature] < split
        left_child_features, left_child_labels = x[row_indices_left_child], y[row_indices_left_child]
        right_child_features, right_child_labels = x[~row_indices_left_child], y[~row_indices_left_child]

        # recursively call build tree of child nodes
        node.left_child = self.__build_tree(left_child_features,
                                            left_child_labels, depth + 1)
        node.right_child = self.__build_tree(right_child_features,
                                             right_child_labels, depth + 1)

        return node

    def __find_best_split(self, x, y):
        """
        Finds the best binary integer split given training data according to
        informatiom gain.
        :param x: numpy.array
                An N by K numpy array (N is the number of instances, K is the
                number of attributes)
        :param y: numpy.array
                An N-dimensional numpy array
        :return:
        Int: An index of the feature used to split the data
        Int: The integer used to split the data (int).
        """
        data = np.transpose(np.vstack((np.transpose(x), y)))
        num_features = data.shape[1] - 1

        # initialise splitting rule components
        integer_splitting_rule = None
        feature_index_to_split = None
        max_info_gain = 0

        # iterate over all the features and find best splits within these
        for feature in range(num_features):
            info_gain, split_int = self.__find_best_split_in_feature(
                data[:, [feature, -1]])
            if info_gain is None:
                continue
            # update max info gain so far as it iterates over features
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                feature_index_to_split = feature
                integer_splitting_rule = int(split_int)

        return feature_index_to_split, integer_splitting_rule

    def __find_best_split_in_feature(self, feature_and_class):
        """
        Finds the best integer split given a certain feature according to
        information gain
        :param feature_and_class: numpy.array
               A N x 2 array containing the feature in the first column and
               class in the second column.
        :return:
        Int: The maximum information gain from a split
        Int: The integer used for that split (int).
        """

        # sort the feature and class and use changes in the class to reduce
        # number of potential split info gain calculations
        sorted_data = feature_and_class[
            feature_and_class[:, 0].astype(np.int).argsort()]
        potential_splits = self.__find_integers_with_class_change(sorted_data)
        info_gains = self.__info_gain_from_splits(potential_splits,
                                                  sorted_data)

        # returning nothing in no information gains are found
        if len(info_gains) == 0:
            return None, None

        index = info_gains.index(max(info_gains))
        return info_gains[index], potential_splits[index]

    def __find_integers_with_class_change(self, sorted_data):
        """
        Finds all the unique integers that correspond to a change in class.
        :param sorted_data: numpy.array
                A N x 2 numpy array containing the feature in the
                first column and class in the second column (sorted in
                ascending order by the feature).
        :return:
        Int List: A unique list of integers for the potential splitting rules.
        """
        potential_integer_splits = []
        for row in range(sorted_data.shape[0] - 1):
            if sorted_data[row, 1] != sorted_data[row + 1, 1]:
                potential_integer_splits.append(sorted_data[row + 1, 0])

        # return only the unique integers that are found
        unique_ints = list(dict.fromkeys(potential_integer_splits))
        return unique_ints

    def __info_gain_from_splits(self, potential_integer_splits, sorted_data):
        """
        Calculates each of the information gains for a list of integer splits
        with the given sorted feature and class data.
        :param potential_integer_splits: Int List
                A list of the potential integer splitting rules.
        :param sorted_data: numpy.array
                A N x 2 numpy array containing the feature in the first
                column and class in the second column (sorted in ascending order
                by the feature).
        :return:
        Int List: A list of the calculated information gains corresponding to
        each of the potential integer splits.
        """
        info_gains = []
        for split in map(int, potential_integer_splits):
            left_child = sorted_data[sorted_data[:, 0].astype(int) < split, :]
            right_child = sorted_data[sorted_data[:, 0].astype(int) >= split, :]
            info_gains.append(self.__calc_info_gain(sorted_data, left_child,
                                                    right_child))
        return info_gains

    def __calc_info_gain(self, parent, left_child, right_child):
        """
        Calculates the information gain given a parent and its two children
        (after a specific split).
        :param parent: numpy.array
                A N x 2 array containing the feature in the first column and
                class in the second column (sorted in ascending order by the
                feature).
        :param left_child: numpy.array
                A N x 2 array containing particular instances of the parent.
        :param right_child: numpy.array
                A N x 2 array containing particular instances of the parent.
        :return:
        Int: The entropy difference between the parent and the children.
        """
        parent_entropy = self.__entropy(parent[:, -1])

        num_rows_left = left_child.shape[0]
        num_rows_right = right_child.shape[0]
        num_rows_total = num_rows_left + num_rows_right

        # don't calculate if any of the children rows are empty
        if num_rows_left == 0 or num_rows_right == 0:
            return 0

        # calculate entropy of the children data
        left_child_entropy = self.__entropy(left_child[:, -1])
        right_child_entropy = self.__entropy(right_child[:, -1])
        left_child_contribution = (num_rows_left/num_rows_total)*left_child_entropy
        right_child_contribution = (num_rows_right/num_rows_total)*right_child_entropy
        new_entropy = left_child_contribution + right_child_contribution

        info_gain = parent_entropy - new_entropy
        return info_gain

    def __entropy(self, labels):
        """
        Calculates the entropy of given data.
        :param feature_and_class: numpy.array
                A N-dimensional array containing the feature in the
                first column and class in the second column (sorted in
                ascending order by the feature).
        :return:
        Int: The calculated entropy value.
        """
        class_probs = np.unique(labels, return_counts=True)[1] / labels.size
        class_prob_logs = np.log2(class_probs)
        entropy = -np.sum(class_probs * class_prob_logs)
        return entropy

    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Assumes that the DecisionTreeClassifier has already been trained.

        Parameters
        ----------
        x : numpy.array
            An N by K numpy array (N is the number of samples, K is the
            number of attributes)

        Returns
        -------
        numpy.array
            An N-dimensional numpy array containing the predicted class label
            for each instance in x
        """

        # make sure that classifier has been trained before predicting
        if not self.is_trained:
            raise Exception(
                "Decision Tree classifier has not yet been trained.")

        # set up empty N-dimensional vector to store predicted labels
        # feel free to change this if needed
        sample_size = x.shape[0]
        predictions = np.zeros((sample_size,), dtype=np.object)

        # map traverse tree function to each row in x
        # convert each instance to int
        # assign that to prediction
        for i in range(sample_size):
            predictions[i] = self.__traverse_tree(self.root, x[i, :])

        # remember to change this if you rename the variable
        return predictions

    def __traverse_tree(self, node, sample_instance):
        """
        Recursively traverses a tree - selecting the either the left or right
        child of a given node until a leaf is found.
        :param node: Node
                A given node (decision rule)
        :param sample_instance: numpy.array
                A 1 x K array representing the features of a particular
                sample instance
        :return: A character prediction for the given sample instance
        """
        if node.is_leaf:
            return node.predicted_class
        split = node.integer_splitting_rule
        feature = node.feature_index_split

        # left node gets assigned to data that is less than the integer
        # splitting rule within that feature
        if sample_instance[feature] < split:
            prediction = self.__traverse_tree(node.left_child,
                                              sample_instance)
        else:
            prediction = self.__traverse_tree(node.right_child,
                                              sample_instance)
        return prediction

    def print_tree(self):
        """
        Prints the decision tree classfier.
        """
        self.__print_node(self.root, 0)

    def __print_node(self, node, depth):
        """
        Recursively prints a node within the tree (up to a depth of four).
        :param node: Node
        :param depth: Int
                    Depth of the node within the tree
        """
        max_depth = 4

        # format label distribution for printing
        unique_labels = np.unique(node.labels, return_counts=True)
        labels = unique_labels[0]
        label_counts = unique_labels[1]
        lab_dists_format = {}
        for lab, cnt in zip(labels, label_counts):
            lab_dists_format[lab] = cnt

        # print the leaf info and go back up the tree
        if node.is_leaf:
            print("    " * (depth + 1), "+", "----" * (depth + 1), "Leaf:",
                  node.predicted_class, " (Entropy:", round(node.entropy, 4),
                  ")", lab_dists_format)
            return

        # go back up the tree if you've reached printing depth limit
        elif depth >= max_depth:
            return

        # print the node info and continue with its childen
        print("    " * (depth + 1), "+", "----" * (depth + 1), "IntNode",
              "[Split: feature " + str(node.feature_index_split) + ":" +
              str(label_names[node.feature_index_split]),
              "< ", str(node.integer_splitting_rule) + "] (Entropy:",
              round(node.entropy, 4), ")", lab_dists_format)

        self.__print_node(node.left_child, depth + 1)
        self.__print_node(node.right_child, depth + 1)

    def __prune_node(self, accuracy, node, validation_data):
        """
        Prunes a node if doing so increases the validation accuracy.
        :param accuracy: The baseline validation accuracy of the model.
        :param node: The node to be pruned.
        :param validation_data: The validation dataset.
        :return:
            The new validation accuracy if node pruned. Otherwise, the baseline
            validation accuracy.
        """
        classes = np.unique(node.labels)
        class_counts = np.unique(node.labels, return_counts=True)[1]
        node.predicted_class = classes[np.argmax(class_counts)]
        node.is_leaf = True

        x_validation = validation_data.features
        y_validation = validation_data.labels

        predictions = self.predict(x_validation)
        evaluator = Evaluator()
        confusion = evaluator.confusion_matrix(predictions, y_validation)
        accuracy_1 = evaluator.accuracy(confusion)

        if accuracy_1 > accuracy:
            return accuracy_1

        node.is_leaf = False
        return accuracy

    def __prune_tree(self, accuracy, node, validation_data, depth):
        """
        Recursively goes down a tree until the node depth matches the input
        depth. If the node's children are leaves, it checks whether it's optimal
        to prune the node.
        :param accuracy: The baseline validation accuracy.
        :param node: The node to be pruned.
        :param validation_data: The validation data set.
        :param depth: The depth at which to prune.
        :return: Returns the new validation accuracy after the pruning process.
        """

        if node.is_leaf:
            return accuracy

        if node.depth == depth:
            if node.left_child.is_leaf and node.right_child.is_leaf:
                accuracy = self.__prune_node(accuracy, node, validation_data)
                return accuracy

        accuracy = self.__prune_tree(accuracy, node.left_child, validation_data, depth)
        accuracy = self.__prune_tree(accuracy, node.right_child, validation_data, depth)

        return accuracy

    def prune(self, accuracy, validation_data):
        """
        Initiates the recursive pruning method starting from the maximum_depth
        of the tree and decreasing the depth by one iteratively.
        :param accuracy: The baseline validation accuracy.
        :param validation_data: The validation data set.
        :return:
            The final validation accuracy after pruning.
        """
        for depth in range(self.maximum_depth, 0, -1):
            accuracy = self.__prune_tree(accuracy, self.root, validation_data, depth)

        return accuracy
