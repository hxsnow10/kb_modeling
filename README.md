KB_modeling
=======
### Objects/applications
* vectorize of relations, nodes, also the arithmic of relation
* 相似点/边查询
* 异常发现；图补齐(包括head, tail , node; 甚至只有1个node的情况下)；
* 推理：
** 问答
** 
* dense vectorize 是信息量减少的过程，做分类是可以的，做一些具体的推理不科学，比如"习近平的爸爸是谁"，向量化不靠谱

### Support
* models: TransE/TransD/TransR/DistmMul/ProjSoftmax  to add: IRN,Conv,
* margin based; sampled sigmoid/softmax.   i prefer second.
* data_loader on WN, FB, Wikidata ( new)
* metrics on (MR, MRR, Hits@N)\*(Raw, Filter)

### TODO1

reimplement:
* TransE, TransD, TransH, TransR, TorusE, 
* DISTMULT, ProjE
* ConvE, ConvKB, IRN
* 

### referneces
* Nguyen, Dat Quoc. “An Overview of Embedding Models of Entities and Relationships for Knowledge Base Completion.” ArXiv:1703.08098 [Cs], March 23, 2017. http://arxiv.org/abs/1703.08098.

### Results

FB15K
distmul  MR_tail=153

