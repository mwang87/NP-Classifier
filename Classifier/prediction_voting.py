import numpy as np


# TODO this can be vastly simplified again


def vote_classification(n_path, n_class, n_super,
                        pred_class, pred_super,
                        path_from_class, path_from_superclass,
                        index):
    """
    Classify the obtained classes

    @param n_path: predicted pathways above noise
    @param n_class: predicted classes above noise
    @param n_super: predicted superclasses above noise
    @param pred_class: predicted classes (all)
    @param pred_super: predicted superclasses (all)
    @param path_from_class: pathways extracted from classes
    @param path_from_superclass: pathways extracted from superclasses
    @param index: the "ontology"
    @return: pathway_result, superclass_result, class_result => the classified results ids
    """

    pathways_for_vote = list(n_path) + list(path_from_class) + list(path_from_superclass)

    # Select pathways with at least 3 counts

    path = {k for k in pathways_for_vote if pathways_for_vote.count(k) == 3}

    if not path:  # if path is empty, we select pathways with only 2 counts
        path = {k for k in pathways_for_vote if pathways_for_vote.count(k) == 2}

    if not path:  # if path is still empty
        # Add all the pathways we already know about, the rest is empty
        return n_path, [], []

    if path & set(n_path):  # we have at least some of n_path in our path
        if path & set(path_from_superclass):  # we have at least some of the path from superclass in path
            n_super = [i for i in n_super if path & set(index['Super_hierarchy'][str(i)]['Pathway'])]
            n_class = [i for i in n_class if path & set(index['Class_hierarchy'][str(i)]['Pathway'])]

            if not n_super:  # n_super is empty
                n_super = list({index['Class_hierarchy'][str(n)]['Superclass'] for n in n_class})

            elif len(n_super) > 1:  # super != []
                if n_class:  # n_class is not empty
                    n_super = list({j for i in n_class for j in index['Class_hierarchy'][str(i)]['Superclass']})
                    n_path = list({j for i in n_class for j in index['Class_hierarchy'][str(i)]['Pathway']})
                elif len(path) == 1:
                    n_super = [np.argmax(pred_super)]
                    best_candidate_class_index = np.argmax(pred_class)
                    if set(n_super) & set(index['Class_hierarchy'][str(best_candidate_class_index)]['Superclass']):
                        n_class = [best_candidate_class_index]

            else:  # we have only one n_super
                # now our classes are only the classes from our best superclass
                n_class = [i for i in n_class if set(n_super) & set(index['Class_hierarchy'][str(i)]['Superclass'])]
                if not n_class:  # n_class is empty, we take the predicted class with the highest score
                    best_candidate_class_index = np.argmax(pred_class)
                    if set(n_super) & set(index['Class_hierarchy'][str(best_candidate_class_index)]['Superclass']):
                        n_class = [best_candidate_class_index]

        else:
            n_class = [p for p in n_class if path & set(index['Class_hierarchy'][str(p)]['Pathway'])]
            n_super = list({index['Class_hierarchy'][str(q)]['Superclass'] for q in n_class})

    else:
        # Select all the classes that are part of our selected pathways
        n_class = [m for m in n_class if set(path) & set(index['Class_hierarchy'][str(m)]['Pathway'])]
        n_super = list({index['Class_hierarchy'][str(n)]['Superclass'] for n in n_class})
        n_path = list({index['Class_hierarchy'][str(v)]['Pathway'] for v in n_class})

    return path, n_super, n_class  # We have to check if we want path or n_path here!
