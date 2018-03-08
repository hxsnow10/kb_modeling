# encoding=utf-8
from random import randint
import numpy as np
from config import config
def read_dict(path):
    a,b,c={},{},{}
    for k,line in enumerate(open(path).readlines()):
        words  = line.strip().split()
        a[words[0]]=k
        b[k]=words[0]
        c[k]=words[-1]
    return a,b,c

class TripleDataset():
    def __init__(self, path, entity2index, relation2index):
        self.path=path
        self.entity_num=len(entity2index)
        self.relation_num=len(relation2index)
        self.entity2index=entity2index
        self.relation2index=relation2index
        self.entitys=[]
        k=0
        for line in open(self.path):
            if k==0:
                k+=1
                continue
            e1,e2,r = line.strip().split()
            self.entitys.append(e1)
            self.entitys.append(e2)
             
    def __iter__(self):
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
            if len(batch)>=100:
                batch_=np.array(batch[:100], dtype=np.int32)
                yield batch_[:,0],batch_[:,1],batch_[:,2],batch_[:,3],batch_[:,4],batch_[:,5] 
                batch=batch[100:]

def load_data(config,mode="train"):
    entity2id, id2entity, eid2name= read_dict(config.entity_path)
    relation2id, id2relation, rid2name= read_dict(config.relation_path)
    print len(entity2id), len(relation2id)
    data=TripleDataset(config.train_path, entity2id , relation2id)
    return data
    
if __name__=="__main__":
    for x in load_data(config):print x
