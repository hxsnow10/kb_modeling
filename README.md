KB_modeling
=======
### Objects/applications
* vectorize of relations, nodes, also the arithmic of relation(more than vector)
* 相似点/边查询
* 异常发现；图补齐；
* 推理
** dense vectorize 是信息量减少的过程，做分类是可以的，做一些具体的推理不科学，比如"习近平的爸爸是谁"，向量化不靠谱
** 这个问题以后再仔细探索。

### Now Support
* models: TransE/TransD/TransR/DistmMul/ProjSoftmax
* margin based; sampled sigmoid/softmax. i prefer second.
* data_loader on WN, FB, Wikidata
* metrics tensorboard show on (MR, MRR, Hits@N)\*(Raw, Filter)\*(all, 1-n, n-1, n-m) 

### TODO
* add Model:IRN, ConvE, ConvKB, IRN
* Retest
* using features
* trainfrom into application
* joint model text and graph
* link graph to text for text-auto-extarct and reasoning

### referneces
* Nguyen, Dat Quoc. “An Overview of Embedding Models of Entities and Relationships for Knowledge Base Completion.” ArXiv:1703.08098 [Cs], March 23, 2017. http://arxiv.org/abs/1703.08098.

### Results

FB15K
transE   MR_tail=105
distmul  MR_tail=153(Do more)


