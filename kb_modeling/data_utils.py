# encoding=utf-8
from random import randint
import numpy as np
from config import config
from collections import defaultdict

def read_dict(path):
    a,b,c={},{},{}
    for k,line in enumerate(open(path).readlines()):
        words  = line.strip().split()
        a[words[0]]=k
        b[k]=words[0]
        c[k]=words[-1]
    return a,b,c

class TripleDataset():
    def __init__(self, path, entity2index, relation2index, all_true_triples, mode='margin'):
        self.path=path
        self.entity_num=len(entity2index)
        self.relation_num=len(relation2index)
        self.entity2index=entity2index
        self.relation2index=relation2index
        self.all_true_triples=all_true_triples
        self.sample_nums=[5,5]
        self.entitys=[]
        k=0
        for line in open(self.path):
            if k==0:
                k+=1
                continue
            e1,e2,r = line.strip().split()
            self.entitys.append(e1)
            self.entitys.append(e2)
        self.mode=mode
    
    def __iter__(self):
        if self.mode=='margin':
            for ele in self.iter1():
                yield ele
        else:
            for ele in self.iter2():
                yield ele
            
    def iter2(self):
        k=0
        batch=[]
        for line in open(self.path):
            if k==0:
                k+=1
                continue
            e1,e2,r = line.strip().split()
            if not (e1 in self.entity2index and e2 in self.entity2index and r in self.relation2index):continue
            e1,e2,r=self.entity2index[e1], self.entity2index[e2], self.relation2index[r]
            batch.append([])
            def add_sample(e1_, e2_, r_, y):
                batch[-1].append([e1_, e2_, r, y])
            k=0
            add_sample(e1,e2,r,1)
            while k<self.sample_nums[-1]:
                r_ =randint(0,self.relation_num-1)
                if (e1,e2,r_) in self.all_true_triples:continue
                else:
                    add_sample(e1,e2,r_,0)
                    k+=1
            
            k=0
            while k<self.sample_nums[0]:
                e1_ =randint(0,self.entity_num-1)
                if (e1_,e2,r) in self.all_true_triples:continue
                else:
                    add_sample(e1_,e2,r,0)
                    k+=1
            
            k=0
            while k<self.sample_nums[0]:
                e2_ =randint(0,self.entity_num-1)
                if (e1,e2_,r) in self.all_true_triples:continue
                else:
                    add_sample(e1,e2_,r,0)
                    k+=1
            
            if len(batch)>=config.batch_size:
                batch_=np.array(batch[:config.batch_size])
                yield batch_[:,:,0],batch_[:,:,1],batch_[:,:,2], batch_[:,:,3]
                batch=batch[config.batch_size:]

    def iter1(self):
        k=0
        batch=[]
        for line in open(self.path):
            if k==0:
                k+=1
                continue
            e1,e2,r = line.strip().split()
            if not (e1 in self.entity2index and e2 in self.entity2index and r in self.relation2index):continue
            e1,e2,r=self.entity2index[e1], self.entity2index[e2], self.relation2index[r]
            while True:
                r_ =randint(0,self.relation_num-1)
                # r_=self.entitys[randint(0,len(self.entitys)-1)]
                if r_!=r:break
            while True:
                e2_=randint(0,self.entity_num-1)
                if e2_!=e2:break
            batch.append([e1,e2,r,e1,e2,r_])
            batch.append([e1,e2,r,e1,e2_,r])
            while True:
                e1_=randint(0,self.entity_num-1)
                if e1_!=e1:break
            batch.append([e1,e2,r,e1_,e2,r])
            if len(batch)>=config.batch_size:
                batch_=np.array(batch[:config.batch_size], dtype=np.int32)
                yield batch_[:,0],batch_[:,1],batch_[:,2],batch_[:,3],batch_[:,4],batch_[:,5] 
                batch=batch[config.batch_size:]
        

def read_graph(paths):
    def dt():
        return defaultdict(set)
    nexts=defaultdict(dt)
    links=defaultdict(dt)
    before=defaultdict(dt)
    triples=set([])
    for path in paths:
        for line in open(path):
            e1,e2,rel=line.strip().split()
            nexts[e1][rel].add(e2)
            links[e1][e2].add(rel)
            before[e2][rel].add(e1)
            triples.add((e1, e2, rel))
    return triples

def load_data(config,mode="train"):
    entity2id, id2entity, eid2name = read_dict(config.entity_path)
    relation2id, id2relation, rid2name = read_dict(config.relation_path)
    all_true_triples = read_graph([config.train_path, config.dev_path, config.test_path])
    print len(entity2id), len(relation2id)
    data=TripleDataset(config.train_path, entity2id , relation2id, all_true_triples, 'margin')
    return data
    
if __name__=="__main__":
    for k,batch in enumerate(load_data(config)):
        print '-'*20+str(k)+'-'*20
        for tensor in batch:
            print np.array(tensor).shape
        print '-'*20+str(k)+'-'*20
