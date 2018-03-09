# encoding=utf-8

class Config(object):

    def __init__(self):
        
        data_dir=self.data_dir="/opt/xia.hong/data/wiki_data/wikidata/wikidata_triples/train"
        data_dir=self.data_dir="./data/FB15K"
        self.train_path=data_dir+"/train2id.txt"
        self.valid_path=data_dir+"/valid2id.txt"
        self.test_path=data_dir+"/test2id.txt"
        self.entity_path=data_dir+"/entity2id.txt"
        self.relation_path=data_dir+"/relation2id.txt"
        self.entity_total=len(open(data_dir+"/entity2id.txt",'r').readlines())
        self.relation_total=len(open(data_dir+"/relation2id.txt",'r').readlines())
        
        self.model="transE"
        self.loadFromData =True
        self.L1_flag = True
        self.hidden_size = 50
        self.trainTimes = 500
        self.margin = 1.0
        
        self.model_dir="./RESULT/model_FB15K_transE_L1_50_QQP"
        self.summary_dir="./RESULT/log_FB15K_transE_L1_50_QQP"
        self.model_path=self.model_dir+'/model'

config=Config()

