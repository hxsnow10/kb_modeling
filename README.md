KB_modeling
=======
### Now Support
* models: TransE/TransD/TransR/DistmMul/ProjSoftmax
* margin based; sampled sigmoid/softmax. i prefer second.
* data_loader on WN, FB, Wikidata
* metrics tensorboard show on (MR, MRR, Hits@N)\*(Raw, Filter) [(all, 1-n, n-1, n-m)]

### Usage
* 修改config里的数据，模型，训练等参数
* python main.py train
* 观察log的tensorboard上述指标
* python main.py test

### data description
├── dev2id.txt  "\t".join([h,t,r]), h,t,r分别是头结点，尾结点，关系的唯一id 
├── entity2id.txt   结点id映射名字，训练非必须，可以看效果
├── relation2id.txt 关系id映射名字
├── test2id.txt 
└── train2id.txt
目前有3种数据 WN, FB, Wikidata
WN，FB在本文件夹里， Wikidata在ftp上 wiki_data/Wikidata_3_kg.zip

### model description
* TransR/TransD/TransE/DistMul 使用了不同的score函数，f(h,r,t)
* 其他部分: 负采样+Max_margin损失函数(对应main.py MarginBasedModel)
* 另外一种选择：使用N个2分类或者1个N分类(对应main.py BasedLogitsModel), [代码需要验证 ]

### TODO
* add Model:IRN, ConvE, ConvKB, IRN
* Retest
* using features
* trainfrom into application
* joint model text and graph
* link graph to text for text-auto-extarct and reasoning

### Objects/applications
* vectorize of relations, nodes, also the arithmic of relation(more than vector)
* 相似点/边查询
* 异常发现；图补齐；
* 推理
** dense vectorize 是信息量减少的过程，做分类是可以的，做一些具体的推理不科学，比如"习近平的爸爸是谁"，向量化不靠谱
** 这个问题以后再仔细探索。


### referneces
* Nguyen, Dat Quoc. “An Overview of Embedding Models of Entities and Relationships for Knowledge Base Completion.” ArXiv:1703.08098 [Cs], March 23, 2017. http://arxiv.org/abs/1703.08098.

### Results

FB15K
transE   MR_tail=105(d=512,L2,margin=1)
transR   MR_tail=104(d=50,L1,margin=1,batch_size=50)
distmul  MR_tail=153(Do more)


