''' Traditional ml methods, including naive bayes, svm and random forest'''
import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from dataset import class_list, test_path, train_path


def get_data(input_path):
    contents, labels = [], []
    with open(input_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            content, label = line.strip().split('\t')
            contents.append(content)
            labels.append(int(label))
    return (contents, labels)


def linear_SVM():
    return Pipeline([
        ('count_vect', CountVectorizer(ngram_range=(1, 3), analyzer="char")),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss="hinge", alpha=1e-3, random_state=1))
    ])


def MNB_Classifier():
    return Pipeline([
        ('count_vec', CountVectorizer(ngram_range=(1, 3), analyzer="char")),
        ('clf', MultinomialNB())
    ])


def Random_Forest():
    return Pipeline([
        ('count_vect', CountVectorizer(ngram_range=(1, 2), analyzer="char")),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1))
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="naive_bayes",
                        help="One of svm, random_forest and naive_bayes.")
    args = parser.parse_args()

    if args.method == "naive_bayes":
        model = MNB_Classifier()
    elif args.method == "svm":
        model = linear_SVM()
    elif args.method == "random_forest":
        model = Random_Forest()
    else:
        raise ValueError

    train_data, train_labels = get_data(train_path)
    test_data, test_labels = get_data(test_path)

    model.fit(train_data, train_labels)

    predicted = model.predict(test_data)

    report = classification_report(test_labels, predicted,
                                   target_names=class_list, 
                                   digits=4, zero_division=0)
    print(report)


if __name__ == "__main__":
    main()
