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
        
        self.model="transR"         # transE/transD/transR/distmMul
        self.loadFromData =True     # 是否从已有模型加载初始化
        self.L1_flag = True         # 是否使用L1正则，否则使用L2正则
        self.hidden_size = 50       # entity/relation向量长度
        self.trainTimes = 500       # 训练的epoch
        self.margin = 1.0           # margin
        self.batch_size=50          # batch_size
        self.start_learning_rate=0.05   # 初始学习率
        self.mini_learning_rate=0.001   # 最小学习率
        self.decay_steps = 30*len(open(self.train_path).readlines())/self.batch_size    
        self.decay_rate = 0.75
        
        self.model_dir="./RESULT/{data}/model_{model}_{size}_L1={L1}_{margin}".format(
            data=self.data_dir.split('/')[-1],
            model=self.model,
            size=self.hidden_size,
            L1=self.L1_flag,
            margin=self.margin
            )   # 输出模型的目录的路径
        self.summary_dir="./RESULT/{data}/log_{model}_{size}_L1={L1}_{margin}".format(
            data=self.data_dir.split('/')[-1],
            model=self.model,
            size=self.hidden_size,
            L1=self.L1_flag,
            margin=self.margin
            )   # 输出log的路径
        self.model_path=self.model_dir+'/model' # 输出模型的路径
                
        self.sess_conf = tf.ConfigProto(
              gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3),
              device_count = {'CPU': 20, 'GPU':0},
              allow_soft_placement=True,
              log_device_placement=False) # tf的session配置
        print self.summary_dir
        self.sess_conf=None
config=Config()

