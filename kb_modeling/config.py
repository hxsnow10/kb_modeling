# encoding=utf-8
import tensorflow as tf

class Config(object):

    def __init__(self):
        
        data_dir=self.data_dir="/opt/xia.hong/data/wiki_data/wikidata/wikidata_triples/train"
        data_dir=self.data_dir="./data/FB15K"
        self.train_path=data_dir+"/train2id.txt"
        self.dev_path=data_dir+"/valid2id.txt"
        self.test_path=data_dir+"/test2id.txt"
        self.entity_path=data_dir+"/entity2id.txt"
        self.relation_path=data_dir+"/relation2id.txt"
        self.entity_total=len(open(data_dir+"/entity2id.txt",'r').readlines())
        self.relation_total=len(open(data_dir+"/relation2id.txt",'r').readlines())
        
        self.model="distmul"
        self.loadFromData =True
        self.L1_flag = True
        self.hidden_size = 50
        self.trainTimes = 500
        self.margin = 2.0
        self.batch_size=100
        self.start_learning_rate=0.05
        self.decay_steps = 5*len(open(self.train_path).readlines())/self.batch_size
        self.decay_rate = 0.75
        
        self.model_dir="./RESULT/{data}/model_{model}_{size}_L1={L1}_{margin}".format(
            data=self.data_dir.split('/')[-1],
            model=self.model,
            size=self.hidden_size,
            L1=self.L1_flag,
            margin=self.margin
            )
        self.summary_dir="./RESULT/{data}/log_{model}_{size}_L1={L1}_{margin}".format(
            data=self.data_dir.split('/')[-1],
            model=self.model,
            size=self.hidden_size,
            L1=self.L1_flag,
            margin=self.margin
            )
        self.model_path=self.model_dir+'/model'

        self.sess_conf = tf.ConfigProto(
              gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3),
              device_count = {'CPU': 20, 'GPU':1},
              allow_soft_placement=True,
              log_device_placement=False)
        # self.sess_conf=None
config=Config()

