import numpy as np


class Evaluator(object):
    """
    Class to perform evaluation.
    """

    def confusion_matrix(self, prediction, annotation, class_labels=None):
        """ Computes the confusion matrix.

        Parameters
        ----------
        prediction : np.array
            an N dimensional numpy array containing the predicted
            class labels
        annotation : np.array
            an N dimensional numpy array containing the ground truth
            class labels
        class_labels : np.array
            a C dimensional numpy array containing the ordered set of class
            labels. If not provided, defaults to all unique values in
            annotation.

        Returns
        -------
        np.array
            a C by C matrix, where C is the number of classes.
            Classes should be ordered by class_labels.
            Rows are ground truth per class, columns are predictions.
        """

        if not class_labels:
            class_labels = np.unique(np.append(annotation, prediction))
        confusion = np.zeros((len(class_labels), len(class_labels)),
                             dtype=np.int)

        for a, p in zip(annotation, prediction):
            a_index = np.where(class_labels == a)
            p_index = np.where(class_labels == p)

            confusion[a_index, p_index] += 1

        return confusion

    def accuracy(self, confusion):
        """ Computes the accuracy given a confusion matrix.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions

        Returns
        -------
        float
            The accuracy (between 0.0 to 1.0 inclusive)
        """
        diagonal = confusion.trace()
        total = confusion.sum()
        accuracy = diagonal / total


        return accuracy

    def precision(self, confusion):
        """ Computes the precision score per class given a confusion matrix.

        Also returns the macro-averaged precision across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the precision score for each
            class in the same order as given in the confusion matrix.
        float
            The macro-averaged precision score across C classes.
        """

        # Initialise array to store precision for C classes
        p = np.zeros((len(confusion),))

        for i in range(0, len(confusion)):
            col_total = confusion[:, i].sum()
            correct = confusion[i, i]

            p[i] = correct / col_total

        macro_p = p.sum() / len(p)
        return p, macro_p

    def recall(self, confusion):
        """ Computes the recall score per class given a confusion matrix.

        Also returns the macro-averaged recall across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the recall score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged recall score across C classes.
        """

        # Initialise array to store recall for C classes
        r = np.zeros((len(confusion),))

        for i in range(0, len(confusion)):
            row_total = confusion[i].sum()
            correct = confusion[i, i]

            r[i] = correct / row_total

        macro_r = r.sum() / len(r)

        return r, macro_r

    def f1_score(self, confusion):
        """ Computes the f1 score per class given a confusion matrix.

        Also returns the macro-averaged f1-score across classes.

        Parameters
        ----------
        confusion : np.array
            The confusion matrix (C by C, where C is the number of classes).
            Rows are ground truth per class, columns are predictions.

        Returns
        -------
        np.array
            A C-dimensional numpy array, with the f1 score for each
            class in the same order as given in the confusion matrix.

        float
            The macro-averaged f1 score across C classes.
        """

        # Initialise array to store recall for C classes
        f = np.zeros((len(confusion),))

        precision = self.precision(confusion)
        recall = self.recall(confusion)

        for i in range(0, len(precision[0])):
            f1 = 2 * (precision[0][i] * recall[0][i]) / (
                        precision[0][i] + recall[0][i])
            f[i] = f1

        macro_f = f.sum() / len(f)

        return f, macro_f
