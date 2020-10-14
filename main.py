import numpy as np


from loading import DataSet
from classification import DecisionTreeClassifier
from eval import Evaluator


if __name__ == "__main__":
    print("---------------------------------------------------")
    print("EXAMPLE RUN OF IMPLEMENTED METHODS")
    print("---------------------------------------------------")

    print("Loading the training dataset...");
    training_data = DataSet("./data/train_full.txt")
    features = training_data.features
    labels = training_data.labels

    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier = classifier.train(features, labels)

    print("Printing the tree")
    classifier.print_tree()

    print("Loading the test set...")
    test_data = DataSet("./data/test.txt")
    x_test = test_data.features
    y_test = test_data.labels

    predictions = classifier.predict(x_test)
    print("Predictions: {}".format(predictions))
    classes = np.unique(y_test)
    print("Evaluating test predictions...")
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, y_test)

    print("Confusion matrix for test data:")
    print(confusion)
    accuracy = evaluator.accuracy(confusion)

    print()
    print("Accuracy: ", accuracy)
    (p, macro_p) = evaluator.precision(confusion)
    (r, macro_r) = evaluator.recall(confusion)
    (f, macro_f) = evaluator.f1_score(confusion)
    print()
    print("Class: Precision, Recall, F1")
    for (i, (p1, r1, f1)) in enumerate(zip(p, r, f)):
        print("{}: {:.2f}, {:.2f}, {:.2f}".format(classes[i], p1, r1, f1));
    print()
    print("Macro-averaged Precision: {:.2f}".format(macro_p))
    print("Macro-averaged Recall: {:.2f}".format(macro_r))
    print("Macro-averaged F1: {:.2f}".format(macro_f))


    print()
    print("---------------------------------------------------")
    print("EXAMPLE RUN OF ACCURACY COMPARISONS ACROSS MODELS")
    print("---------------------------------------------------")


    validation_data = DataSet("./data/validation.txt")
    x_validation = validation_data.features
    y_validation = validation_data.labels
    predictions = classifier.predict(x_validation)
    classes = np.unique(y_validation)
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, y_validation)
    validation_accuracy_non_pruned_full = evaluator.accuracy(confusion)
    max_depth_non_pruned_full = classifier.maximum_depth
    validation_accuracy_pruned_full = classifier.prune(validation_accuracy_non_pruned_full, validation_data)
    max_depth_pruned_full=classifier.maximum_depth
    predictions = classifier.predict(x_test)
    classes = np.unique(y_test)
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, y_test)
    test_accuracy_pruned_full = evaluator.accuracy(confusion)

    training_data = DataSet("./data/train_noisy.txt")
    features = training_data.features
    labels = training_data.labels

    classifier = DecisionTreeClassifier()
    classifier = classifier.train(features, labels)
    max_depth_non_pruned_noisy=classifier.maximum_depth

    predictions = classifier.predict(x_test)
    classes = np.unique(y_test)
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, y_test)
    test_accuracy_non_pruned_noisy = evaluator.accuracy(confusion)


    predictions = classifier.predict(x_validation)
    classes = np.unique(y_validation)
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, y_validation)
    validation_accuracy_non_pruned_noisy = evaluator.accuracy(confusion)
    validation_accuracy_pruned_noisy = classifier.prune(validation_accuracy_non_pruned_noisy, validation_data)
    max_depth_pruned_noisy=classifier.maximum_depth

    predictions = classifier.predict(x_test)
    classes = np.unique(y_test)
    evaluator = Evaluator()
    confusion = evaluator.confusion_matrix(predictions, y_test)
    test_accuracy_pruned_noisy = evaluator.accuracy(confusion)

    print("For full_train.txt:")
    print("Test accuracy non pruned tree: ",test_accuracy_non_pruned_full)
    print("Test accuracy pruned tree: ",test_accuracy_pruned_full)
    print("validation accuracy non pruned tree: ",validation_accuracy_non_pruned_full)
    print("validation accuracy pruned tree: ",validation_accuracy_pruned_full)
    print()

    print("For noisy_train.txt:")
    print("Test accuracy non pruned tree: ",test_accuracy_non_pruned_noisy)
    print("Test accuracy pruned tree: ",test_accuracy_pruned_noisy)
    print("validation accuracy non pruned tree: ",validation_accuracy_non_pruned_noisy)
    print("validation accuracy pruned tree: ",validation_accuracy_pruned_noisy)
    print()

    print("For full_train.txt:")
    print("max depth non-pruned tree",max_depth_non_pruned_full)
    print("max depth pruned tree",max_depth_pruned_full)
    print()

    print("For noisy_train.txt:")
    print("max depth non-pruned tree",max_depth_non_pruned_noisy)
    print("max depth pruned tree",max_depth_pruned_noisy)
    print()
    
  
