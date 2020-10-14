from scipy.stats import *
import numpy as np
from loading import DataSet
from classification import DecisionTreeClassifier
from eval import Evaluator

def split_into_folds(features, labels, folds):
    '''
    Splits the training data into a given number of folds.
    :param features: numpy.array
            A N x K array of the features of the training data.
    :param labels: numpy.array
            A N x 1 array of the labels of the training data.
    :param folds: Int
            The number of folds to split the data into.
    :return:
            The training data split into multiple folds.
    '''
    data = np.transpose(np.vstack((np.transpose(features), labels)))
    np.random.shuffle(data)
    split_data = np.array_split(data, folds)
    return split_data


def cross_validation(features, labels, folds):
    '''
    Performs cross validation process.
    :param features: numpy.array
            A N x K array of the features of the training data.
    :param labels: numpy.array
            A N x 1 array of the labels of the training data.
    :param folds: Int
            The number of folds to split the data into.
    :return:
        A list of the trained classifier models.
        A list of all their  predictions.
        A list of all their accuracies.
    '''
    accuracies = []
    classifiers = []
    predictions_list = []

    split_data = split_into_folds(features, labels, folds)

    for i in range(folds):
        test_data = split_data[i]
        train_folds = []
        for j in range(folds):
            if j != i:
                train_folds.append(split_data[j])

        flatten = lambda l: [item for sublist in l for item in sublist]
        train_data = np.vstack(flatten(train_folds))

        train_features_str = train_data[0:, :-1]
        train_features = train_features_str.astype(np.int)
        train_labels = np.transpose(train_data[0:, -1])

        test_features_str = test_data[0:, :-1]
        test_features = test_features_str.astype(np.int)
        test_labels = np.transpose(test_data[0:, -1])

        classifier = DecisionTreeClassifier()
        classifiers.append(classifier)

        classifier = classifier.train(np.array(train_features), np.array(train_labels))

        evaluator = Evaluator()
        predictions = classifier.predict(test_features)
        confusion = evaluator.confusion_matrix(predictions, test_labels)
        accuracy = evaluator.accuracy(confusion)

        predictions_list.append(predictions)
        accuracies.append(accuracy)

    return classifiers, predictions_list, accuracies


def main():
    training_data = DataSet("./data/train_full.txt")
    x_train = training_data.features
    y_train = training_data.labels

    test_data = DataSet("./data/test.txt")
    x_test = test_data.features
    y_test = test_data.labels
    classes = np.unique(y_test)

    print("For the original un-pruned tree...")
    classifier = DecisionTreeClassifier()
    classifier = classifier.train(x_train, y_train)
    predictions = classifier.predict(x_test)
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, y_test)

    print("Confusion matrix:")
    print(confusion)
    accuracy = evaluator.accuracy(confusion)
    print()
    print("Accuracy: {}".format(accuracy))
    (p, macro_p) = evaluator.precision(confusion)
    (r, macro_r) = evaluator.recall(confusion)
    (f, macro_f) = evaluator.f1_score(confusion)
    print()
    print("Class: Precision, Recall, F1")
    for (i, (p1, r1, f1)) in enumerate(zip(p, r, f)):
        print("{}: {:.2f}, {:.2f}, {:.2f}".format(classes[i], p1, r1, f1));

    folds = 10
    classifiers, predictions, accuracies = cross_validation(x_train, y_train, folds)
    accuracies = np.array(accuracies)
    avg_accuracy = np.average(accuracies)
    std = np.std(accuracies)
    print("Using cross validation with 10 folds: the average accuracy is %.4f "
          "and the standard deviation is %.4f" %(avg_accuracy, std))


    print("For the best classifier...")
    best_classifier = classifiers[np.argmax(accuracies)]
    best_classifier.predict(x_test)
    predictions = best_classifier.predict(x_test)
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, y_test)

    print("Confusion matrix:")
    print(confusion)

    accuracy = evaluator.accuracy(confusion)

    print()
    print("Accuracy: {}".format(accuracy))
    (p, macro_p) = evaluator.precision(confusion)
    (r, macro_r) = evaluator.recall(confusion)
    (f, macro_f) = evaluator.f1_score(confusion)

    print()
    print("Class: Precision, Recall, F1")
    for (i, (p1, r1, f1)) in enumerate(zip(p, r, f)):
        print("{}: {:.2f}, {:.2f}, {:.2f}".format(classes[i], p1, r1, f1));
    print()
    print("Macro-averaged Precision: {:.3f}".format(macro_p))
    print("Macro-averaged Recall: {:.3f}".format(macro_r))
    print("Macro-averaged F1: {:.3f}".format(macro_f))


    all_predictions = []
    for classifier in classifiers:
        classifier.predict(x_test)
        predictions = classifier.predict(x_test)
        all_predictions.append(predictions)

    print("Taking the mode of all classifiers' votes...")
    votes = np.vstack(all_predictions)
    final_votes = np.array(mode(votes)[0][0]) #  using scipy.stats
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(final_votes, y_test)
    print("Confusion matrix:")
    print(confusion)
    accuracy = evaluator.accuracy(confusion)
    print()
    print("Accuracy: {}".format(accuracy))
    (p, macro_p) = evaluator.precision(confusion)
    (r, macro_r) = evaluator.recall(confusion)
    (f, macro_f) = evaluator.f1_score(confusion)
    print()
    print("Class: Precision, Recall, F1")
    for (i, (p1, r1, f1)) in enumerate(zip(p, r, f)):
        print("{}: {:.2f}, {:.2f}, {:.2f}".format(classes[i], p1, r1, f1));
    print()
    print("Macro-averaged Precision: {:.3f}".format(macro_p))
    print("Macro-averaged Recall: {:.3f}".format(macro_r))
    print("Macro-averaged F1: {:.3f}".format(macro_f))


if __name__ == "__main__":
    main()
