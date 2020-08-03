
import itertools 
import numpy as np

def vote_classification(n_path, 
                        n_class, 
                        n_super, 
                        pred_class, 
                        pred_super, 
                        path_from_class, 
                        path_from_superclass, isglycoside, ontology_dictionary):
    class_result = []
    superclass_result = []
    pathway_result = []

    index = ontology_dictionary

    index_class = list(index['Class'].keys())
    index_superclass = list(index['Superclass'].keys())
    index_pathway = list(index['Pathway'].keys())

    path_for_vote = n_path+path_from_class+path_from_superclass
    path = list(set([ k for k in path_for_vote if path_for_vote.count(k) ==3]))

    if path == []:
        path = list(set([ k for k in path_for_vote if path_for_vote.count(k) ==2]))
        if len(path)>1:
            path = list(set([ k for k in path_for_vote if path_for_vote.count(k) ==2])) 
    if path == []:
        for w in n_path:
            pathway_result.append(index_pathway[w])
        return pathway_result,superclass_result,class_result,isglycoside


    else: #path != []
        if set(n_path) & set(path) != set():
            if set(path) & set(path_from_superclass) != set():
                n_super = [ l for l in n_super if set(path)& set(index['Super_hierarchy'][str(l)]['Pathway']) != set()]
                if n_super == []:
                    n_class = [ m for m in n_class if set(path) & set(index['Class_hierarchy'][str(m)]['Pathway']) != set() ]
                    n_super = [index['Class_hierarchy'][str(n)]['Superclass'] for n in n_class]
                    n_super = list(set(itertools.chain.from_iterable(n_super)))

                elif len(n_super) > 1: #super != []
                    n_class = [ u for u in n_class if set(path) & set(index['Class_hierarchy'][str(u)]['Pathway']) != set() ]
                    if n_class != []:
                        n_super = [index['Class_hierarchy'][str(v)]['Superclass'] for v in n_class]
                        n_path = [index['Class_hierarchy'][str(v)]['Pathway'] for v in n_class]
                        n_path = list(set(itertools.chain.from_iterable(n_path)))
                        n_super = list(set(itertools.chain.from_iterable(n_super)))

                    elif len(path)==1:
                        n_super = [np.argmax(pred_super)]
                        n_class = [ m for m in [np.argmax(pred_class)] if set(n_super) & set(index['Class_hierarchy'][str(m)]['Superclass']) != set() ]


                else:
                    n_class = [ o for o in n_class if set(n_super) & set(index['Class_hierarchy'][str(o)]['Superclass'])!=set() ]
                    if n_class == []:
                        n_class = [ m for m in [np.argmax(pred_class)] if set(n_super) & set(index['Class_hierarchy'][str(m)]['Superclass']) != set() ]
            else:
                n_class = [ p for p in n_class if  set(path) & set(index['Class_hierarchy'][str(p)]['Pathway']) !=set() ]
                n_super = [index['Class_hierarchy'][str(q)]['Superclass'] for q in n_class]

                n_super = list(set(itertools.chain.from_iterable(n_super)))

        else:
            n_super = [ l for l in n_super if set(path) & set(index['Super_hierarchy'][str(l)]['Pathway']) != set()]
            if n_super == []:
                n_class = [ m for m in n_class if set(path) & set(index['Class_hierarchy'][str(m)]['Pathway']) != set()]
                n_super = [index['Class_hierarchy'][str(n)]['Superclass'] for n in n_class]
                n_path = [index['Class_hierarchy'][str(v)]['Pathway'] for v in n_class]
                n_path = list(set(itertools.chain.from_iterable(n_path)))
                n_super = list(set(itertools.chain.from_iterable(n_super)))


            elif len(n_super) > 1: #super != []
                n_class = [ u for u in n_class if set(path) & set(index['Class_hierarchy'][str(u)]['Pathway']) != set()]
                n_super = [index['Class_hierarchy'][str(v)]['Superclass'] for v in n_class]
                n_path = [index['Class_hierarchy'][str(v)]['Pathway'] for v in n_class]
                n_path = list(set(itertools.chain.from_iterable(n_path)))
                n_super = list(set(itertools.chain.from_iterable(n_super)))


            else:
                n_class = [ o for o in n_class if set(path) & set(index['Class_hierarchy'][str(o)]['Pathway']) != set() ]
                n_super = [index['Class_hierarchy'][str(v)]['Superclass'] for v in n_class]
                n_path = [index['Class_hierarchy'][str(v)]['Pathway'] for v in n_class]
                n_path = list(set(itertools.chain.from_iterable(n_path)))
                n_super = list(set(itertools.chain.from_iterable(n_super)))

    for r in path:
        pathway_result.append(index_pathway[r])
    for s in n_super:
        superclass_result.append(index_superclass[s])
    for t in n_class:
        class_result.append(index_class[t])
    
    return pathway_result,superclass_result,class_result,isglycoside #three class results and glycoside checker result (True/False)
    

