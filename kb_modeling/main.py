# encoding=utf-8
import tensorflow as tf
from data_utils import load_data, read_dict, read_graph
from tf_utils import load_config
from tf_utils import check_dir
from shutil import copy
import heapq
config=load_config()
from model import TransEModel, TransRModel, TransDModel
import sys
 
model={
    "transE": TransEModel,
    "transD": TransRModel,
    "transR": TransDModel
}

def evaluate(sess, trainModel, data_path):
    entity2id, id2entity, eid2name= read_dict(config.entity_path)
    relation2id, id2relation, rid2name= read_dict(config.relation_path)
    true_triples=read_graph([config.train_path, config.test_path])
    true_triples=set([(entity2id[h], entity2id[t], relation2id[r]) for h,t,r in true_triples])
    # (Raw, Filtered) * (MR, Hits@10, Hits@5, Hits@1, MRR)
    # Raw: all corrupted triple;  Filtered: all-true triple
    # MR: ranking of test_triple
    # Hits@N: whether test_triple in topN
    # MRR: 1/MR
    # everage -> final
    
    def test_step(pos_h, pos_t, pos_r):
        feed_dict = {
            trainModel.pos_h: [pos_h],
            trainModel.pos_t: [pos_t],
            trainModel.pos_r: [pos_r],
        }
        predict = sess.run(
            [trainModel.predict], feed_dict)
        return -predict
    
    def rank(test_triple, compare_triples, filtered=False):
        s=len(compare_triples)
        if filtered:
            compare_triples=[triple for triple in compare_triples if triple not in true_triples]
        dist=test_step(*test_triple)
        dists=[test_step(*triple) for triple in compare_triples]

        print '1'
        MR=0
        for k,dist_ in enumerate(dists):
            if compare_triples[k]==test_triple: continue
            if dist_>dist:
                # if rank<=10: print k, get_name(compare_triples[k])
                MR+=1
        MR+=1
        MRR=1.0/MR
        samples = zip(compare_triples, dists)
        print '2'
        top10=heapq.nlargest(10,samples, key=lambda x:x[1])
        top10=[x[0] for x in top10]
        top5=top10[:5]
        top1=top10[:1]
        hits10=1 if test_triple in top10 else 0
        hits5=1 if test_triple in top5 else 0
        hits1=1 if test_triple in top1 else 0
        metrics=[MR, MRR, hits10, hits5, hits1]
        return metrics
    metrics=[] 
    for line in open(data_path).readlines():
        try:
            h, t ,r = line.strip().split()
            h,t,r = entity2id[h], entity2id[t], relation2id[r]
            print h,t,r
            print eid2name[h],eid2name[t],rid2name[r]
            compare_triples1 = [(h, t, r_) for r_ in id2relation.keys()]
            compare_triples2 = [(h_, t, r) for h_ in id2entity.keys()]
            compare_triples3 = [(h, t_, r) for t_ in id2entity.keys()]
            metrics1 = [rank( (h,t,r), compare_triples1), rank( (h,t,r), compare_triples1, True)]
            metrics2 = [rank( (h,t,r), compare_triples2), rank( (h,t,r), compare_triples2, True)]
            metrics3 = [rank( (h,t,r), compare_triples3), rank( (h,t,r), compare_triples3, True)]
            print [metrics1, metrics2, metrics3]
            metrics.append([metrics1, metrics2, metrics3])
        except Exception,e:
            import traceback
            traceback.print_exc()
    metrics=np.mean(np.array(metrics),0)
    return metrics

def main(_):
    config = load_config()
    
    Model = model[config.model]
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer(uniform = False)
            with tf.variable_scope("model", reuse=None, initializer = initializer):
                trainModel = Model(config = config)

            sess.run(trainModel.init)
            
            if (config.loadFromData) or sys.argv[1]=="test":
                trainModel.saver.restore(sess, config.model_path)
            
            def train_step(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
                feed_dict = {
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                    trainModel.neg_h: neg_h_batch,
                    trainModel.neg_t: neg_t_batch,
                    trainModel.neg_r: neg_r_batch
                }
                _, loss = sess.run(
                    [trainModel.train_op, trainModel.loss], feed_dict)
                return loss


            if sys.argv[1]=="train":
                if  not config.loadFromData:
                    check_dir(config.summary_dir)
                    check_dir(config.model_dir)
                    copy("config.py", config.summary_dir) 
                    copy("config.py", config.model_dir)
                data=load_data(config,"train")
                for times in range(config.trainTimes):
                    res = 0.0
                    for k,batch_data in enumerate(data):
                        ph, pt, pr, nh, nt, nr = batch_data
                        res += train_step(ph, pt, pr, nh, nt, nr)
                        if k%1000==0:
                            print k, res/k
                    # evaluate(sess, trainModel, config.test_path)
                    trainModel.saver.save(sess, config.model_path)
            else:
                evaluate(sess, trainModel, config.test_path)

if __name__ == "__main__":
    tf.app.run()
