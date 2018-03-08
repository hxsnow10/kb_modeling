# encoding=utf-8

class Config(object):

    def __init__(self):
        data_dir=self.data_dir="/opt/xia.hong/data/wiki_data/wikidata/wikidata_triples/train"
        data_dir=self.data_dir="./data/FB15K"
        self.loadFromData = False
        self.L1_flag = True
        self.hidden_size = 50
        self.trainTimes = 500
        self.margin = 1.0
        self.train_path=data_dir+"/train2id.txt"
        self.entity_path=data_dir+"/entity2id.txt"
        self.relation_path=data_dir+"/relation2id.txt"
        self.test_path=data_dir+"/test2id.txt"
        self.entity_total=len(open(data_dir+"/entity2id.txt",'r').readlines())
        self.relation_total=len(open(data_dir+"/relation2id.txt",'r').readlines())
        self.model_path="./FB15K_model/transE_L1_50_QQP/model"

config=Config()

