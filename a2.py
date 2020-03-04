import argparse
import random
from collections import Counter
from sklearn.base import is_classifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import sklearn.metrics as metrics
import math
import numpy as np
import string
random.seed(42)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def part1(samples):
    #extract features
    X = extract_features(samples)
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[0])
    print("Data shape: ", X.shape)
    return X


def extract_features(samples):
    print("Extracting features ...")
    docs_list = []
    all_words_index = {}
    index = 0
    for sample in samples:
        text = sample.lower().split() # lowercase so the words are not treated differently if they appear at the beginning of a sentence.
        text = ["".join(c for c in word if c not in string.punctuation) for word in text if not word.isnumeric()] # removes punctuation marks and numbers
        for word in text:
            if word not in all_words_index:
                all_words_index[word] = index # assigns a unique index to every word in the samples
                index += 1
        text_dict = Counter(text)
        text_dict = {all_words_index[word]: text_dict[word] for word in text_dict} # changes the keys to the relative index in the dictionary, so they can be used in the numpy array
        docs_list.append(text_dict)
    features_array = np.zeros((len(docs_list), len(all_words_index))) # creates empty array of the desired shape (one row for each document, one column for each word)
    doc_index = 0 # creates an index for the rows
    for doc in docs_list:
        indexes = tuple(doc.keys())
        features_array[doc_index][np.array(indexes)] = tuple(doc.values()) # uses advanced indexing to assign the values to the correct index
        doc_index += 1
    print("Done")
    print("Removing unfrequent words ...")
    total_docs = features_array.shape[0]
    features_array = features_array[:, np.sum(features_array > 0, axis=0) < (total_docs / 2)] # Exclude the words that appear in more than half of the documents (probably stop-words)
    features_array = features_array[:, np.any(features_array > 5, axis =0)] # Exclude the words that do not appear more than 5 times in any of the document (unfrequent words)
    print("Number of words:", features_array.shape[1])
    return features_array


def part2(X, n_dim):
    #Reduce Dimension
    print("Reducing dimensions ... ")
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X[0])
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr


def reduce_dim(X,n=10):
    pca = PCA(n_components=n)
    return pca.fit_transform(X)


def get_classifier(clf_id):
    if clf_id == 1:
        clf = SVC(gamma='scale')
    elif clf_id == 2:
        clf = DecisionTreeClassifier()
    else:
        raise KeyError("No clf with id {}".format(clf_id))

    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf


def part3(X, y, clf_id):
    X_train, X_test, y_train, y_test = shuffle_split(X,y)

    #get the model
    clf = get_classifier(clf_id)

    #printing some stats
    print()
    print("Train example: ", X_train[0])
    print("Test example: ", X_test[0])
    print("Train label example: ", y_train[0])
    print("Test label example: ", y_test[0])
    print()


    #train model
    print("Training classifier ...")
    train_classifer(clf, X_train, y_train)


    # evaluate model
    print("Evaluating classifier ...")
    evaluate_classifier(clf, X_test, y_test)


def shuffle_split(X,y):
    indexes_list = list(range(X.shape[0])) # Creates a list of indexes that can be shuffled without losing the parallelism between X and y
    random.shuffle(indexes_list)
    last_training_index = int(0.80 * X.shape[0]) # Calculates the last index to separate the data into two parts with ratio 80/20
    indexes_train = indexes_list[:last_training_index] # Chooses which indexes of the shuffled list will become part of the training or test set
    indexes_test = indexes_list[last_training_index:]
    X_train = X[indexes_train] # Creates 4 different arrays using advanced indexing
    X_test = X[indexes_test]
    y_train = y[indexes_train]
    y_test = y[indexes_test]
    return X_train, X_test, y_train, y_test


def train_classifer(clf, X, y):
    assert is_classifier(clf)
    clf.fit(X, y)


def evaluate_classifier(clf, X, y):
    assert is_classifier(clf)
    y_pred = clf.predict(X)
    print("Accuracy:", metrics.accuracy_score(y, y_pred))
    print("Precision:", metrics.precision_score(y, y_pred, average='weighted'))
    print("Recall:", metrics.recall_score(y, y_pred, average='weighted'))
    print("F-measure:", metrics.f1_score(y, y_pred,average='weighted'))


def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    print("Example data sample:\n\n", data.data[0])
    print("Example label id: ", data.target[0])
    print("Example label name: ", data.target_names[data.target[0]])
    print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names


def part_bonus(X, n_dim):
    #Reduce Dimension
    print("Reducing dimensions ... ")
    svd = TruncatedSVD(n_components=n_dim)
    X_dr = svd.fit_transform(X)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X[0])
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr


#DONT CHANGE THIS FUNCTION
def main(model_id=None, n_dim=False, alternative_reduction=False):

    # load data
    samples, labels, label_names = load_data()


    #PART 1
    print("\n------------PART 1-----------")
    X = part1(samples)

    #part 2
    if n_dim:
        if not alternative_reduction:
            print("\n------------PART 2-----------")
            X = part2(X, n_dim)
        else:
            print("\n------------PART 2-----------")
            X = part_bonus(X, n_dim)

    #part 3
    if model_id:
        print("\n------------PART 3-----------")
        part3(X, labels, model_id)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the features for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    parser.add_argument("-r", "--alternative_reduction", nargs='?', help="use alternative model for dimensionality reduction")
        
    args = parser.parse_args()
    main(model_id=args.model_id, n_dim=args.number_dim_reduce, alternative_reduction=args.alternative_reduction)
