from imblearn.over_sampling import SMOTE

def add_smote_classifiers_to_dict(clf_lst, classifiers):
    for clf in clf_lst:
        smote_clf = clf + "_smote"
        if smote_clf not in classifiers:
            classifiers[smote_clf]["clf_desc"] = classifiers[clf]["clf_desc"]
            classifiers[smote_clf]["model"] = classifiers[clf]["model"]
            classifiers[smote_clf]["c"] = classifiers[clf]["c"]
            classifiers[smote_clf]["cmap"] = classifiers[clf]["cmap"]
            classifiers[smote_clf]["threshold"] = classifiers[clf]["threshold"]
    return classifiers

