# encoding=utf-8
from collections import defaultdict
from copy import deepcopy

def find_different_paths(graph_file_path, entity_file_path=None, relation_file_path=None):
    def dt():
        return defaultdict(int)
    nexts=defaultdict(dt)
    for line in open(graph_file_path).readlines()[1:]:
        e1, e2, rel = line.strip().split()
        nexts[e1][rel]=e2
     
    id2e={}
    for line in open(entity_file_path).readlines()[1:]:
        e, id_ = line.strip().split()
        id2e[id_]=e
    
    id2r={}
    for line in open(entity_file_path).readlines()[1:]:
        r, id_ = line.strip().split()
        id2e[id_]=r

    def path():
        return defaultdict(set)
    paths=defaultdict(path)
    for e1 in nexts:
        for r in nexts[e1]:
            e2=nexts[e1][r]
            paths[e1][e2].add((r,))
    for k in range(1):
        paths2=defaultdict(path)
        for e1 in paths:
            for e2 in paths[e1]:
                for p in paths[e1][e2]:
                    paths2[e1][e2].add(p)
        L, keys=len(paths), paths.keys()
        for e1 in paths:
            for e2 in paths[e1]:
                if e2 not in paths:continue
                for e3 in paths[e2]:
                    if e1==e2 or e2==e3:continue
                    for p1 in paths[e1][e2]:
                        for p2 in paths[e2][e3]:
                            paths2[e1][e3].add(p1+p2)
        paths=paths2
    
    similar_paths=defaultdict(int)
    for e1 in paths:
        for e2 in paths:
            if len(paths[e1][e2])>1:
                for p1 in paths[e1][e2]:
                    for p2 in paths[e1][e2]:
                        if p1!=p2 and len(p1)!=len(p2):
                            similar_paths[(p1,p2)]+=1
    import heapq
    paths = heapq.nlargest(100, similar_paths.iteritems(), key=lambda x:x[1])
    oo=open("paths.txt",'w')
    for p,_ in paths:
        p1,p2=p
        oo.write(str(p1)+'\t'+str(p2)+'\n')
    
find_different_paths("data/FB15K/train2id.txt", "data/FB15K/entity2id.txt", "data/FB15K/relation2id.txt")
