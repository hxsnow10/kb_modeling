# encoding=utf-8
import tensorflow as tf
from data_utils import load_data, read_dict, read_graph
from tf_utils import load_config
from tf_utils import check_dir
from shutil import copy
import heapq
config=load_config()
from model import TransEModel, TransRModel, TransDModel, DistMul
import sys, os
import numpy as np
 
getmodel={
    "transE": TransEModel,
    "transD": TransDModel,
    "transR": TransRModel,
    "distmul": DistMul,
}

def evaluate(sess, trainModel, data_path, n=20):
    entity2id, id2entity, eid2name= read_dict(config.entity_path)
    relation2id, id2relation, rid2name= read_dict(config.relation_path)
    true_triples=read_graph([config.train_path, config.dev_path, config.test_path])
    true_triples=set([(entity2id[h], entity2id[t], relation2id[r]) for h,t,r in true_triples])
    # (Raw, Filtered) * (MR, Hits@10, Hits@5, Hits@1, MRR)
    # Raw: all corrupted triple;  Filtered: all-true triple
    # MR: ranking of test_triple
    # Hits@N: whether test_triple in topN
    # MRR: 1/MR
    # everage -> final
    
    def test_step(pos_h, pos_t, pos_r):
        feed_dict = {
            trainModel.pos_h: pos_h,
            trainModel.pos_t: pos_t,
            trainModel.pos_r: pos_r,
        }
        predict = sess.run(
            [trainModel.predict], feed_dict)
        return -predict[0]
    
    def rank(test_triple, compare_triples, filtered=False):
        s=len(compare_triples)
        if filtered:
            compare_triples=[triple for triple in compare_triples if triple not in true_triples or triple==test_triple]
        
        h,t,r=test_triple
        dist=test_step([h],[t],[r])
        i,dists=0,[]
        while i<len(compare_triples):
            batch=compare_triples[i:i+100]
            batch=[[triple[k] for triple in batch] for k in range(3)]
            batch_h, batch_r, batch_t = batch
            batch_dists = test_step(batch_h, batch_r, batch_t)
            dists = dists + batch_dists.tolist()
            i=i+100

        MR=0
        for k,dist_ in enumerate(dists):
            if compare_triples[k]==test_triple: continue
            if dist_>dist:
                # if rank<=10: print k, get_name(compare_triples[k])
                MR+=1
        MR+=1
        MRR=1.0/MR
        samples = zip(compare_triples, dists)
        top10=heapq.nlargest(10,samples, key=lambda x:x[1])
        top10=[x[0] for x in top10]
        top5=top10[:5]
        top1=top10[:1]
        hits10=1.0 if test_triple in top10 else 0
        hits5=1.0 if test_triple in top5 else 0
        hits1=1.0 if test_triple in top1 else 0
        metrics=[MR, MRR, hits10, hits5, hits1]
        return metrics
    metrics=[]
    for k,line in enumerate(open(data_path).readlines()):
        if k>n:break
        try:
            h, t ,r = line.strip().split()
            h,t,r = entity2id[h], entity2id[t], relation2id[r]
            # print h,t,r
            # print eid2name[h],eid2name[t],rid2name[r]
            compare_triples0 = [(h, t, r_) for r_ in id2relation.keys()]
            compare_triples1 = [(h, t_, r) for t_ in id2entity.keys()]
            compare_triples2 = [(h_, t, r) for h_ in id2entity.keys()]
            metrics0 = [rank( (h,t,r), compare_triples0,True)]#, rank( (h,t,r), compare_triples0) ]
            metrics1 = [rank( (h,t,r), compare_triples1,True)]#, rank( (h,t,r), compare_triples1) ]
            # metrics2 = [rank( (h,t,r), compare_triples2,True), rank( (h,t,r), compare_triples2) ]
            # print [metrics1, metrics2, metrics3]
            metrics.append([metrics0, metrics1])
        except Exception,e:
            import traceback
            traceback.print_exc()
        if k>0 and k%10==0:
            print " [MR, MRR, hits10, hits5, hits1].filter_average_rel=", np.mean(np.array(metrics),0)[0,0,:].tolist()
            print " [MR, MRR, hits10, hits5, hits1].filter_average_head=", np.mean(np.array(metrics),0)[1,0,:].tolist()
            # print " [MR, MRR, hits10, hits5, hits1].filter_average_tail=", np.mean(np.array(metrics),0)[2,1,:].tolist()
    metrics=np.mean(np.array(metrics),0)
    metrics={"rel_Mr":metrics[0,0,0], "rel_hits10":metrics[0,0,2], "rel_hits1":metrics[0,0,4],\
            "tail_Mr":metrics[1,0,0], "tail_hits10":metrics[1,0,2], "tail_hits1":metrics[1,0,4]}
    print metrics
    return metrics

def main(_):
    config = load_config()
    
    Model = getmodel[config.model]
    with tf.Graph().as_default():
        sess = tf.Session(config=config.sess_conf)
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer(uniform = False)
            #with tf.variable_scope("model", reuse=None, initializer = initializer):
            model=trainModel = Model()

            sess.run(trainModel.init)
            
            if (config.loadFromData and sys.argv[1]=="train") or sys.argv[1]=="test":
                try:
                    trainModel.saver.restore(sess, config.model_path)
                except:
                    print "loading error"
            
            def train_data(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
                feed_dict = {
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                    trainModel.neg_h: neg_h_batch,
                    trainModel.neg_t: neg_t_batch,
                    trainModel.neg_r: neg_r_batch
                }
                return feed_dict


            if sys.argv[1]=="train":
                 
                if  not config.loadFromData:
                    check_dir(config.summary_dir)
                    check_dir(config.model_dir)
                    copy("config.py", config.summary_dir) 
                    copy("config.py", config.model_dir)
                    step=0
                else:
                    step=sess.run(model.global_step)
                summary_writers = { 
                    sub_path:tf.summary.FileWriter(os.path.join(config.summary_dir,sub_path), sess.graph, flush_secs=5)
                    for sub_path in ['train','dev','test']}
                data=load_data(config,"train")
                for times in range(config.trainTimes):
                    res = 0.0
                    for k,batch_data in enumerate(data):
                        fd = train_data(*batch_data)
                        if k%1000!=0:
                            loss,_=sess.run([model.loss, model.train_op], feed_dict=fd)
                        else:
                            loss,_,summary=sess.run([model.loss, model.train_op, model.step_summaries],
                                feed_dict=fd)
                            summary_writers['train'].add_summary(summary, step)
                            print k,loss
                        step+=1
                    train_metrics=evaluate(sess, trainModel, config.train_path)
                    dev_metrics=evaluate(sess, trainModel, config.dev_path)
                    test_metrics=evaluate(sess, trainModel, config.test_path)
                    def add_summary(writer, metric, step):
                        for name,value in metric.iteritems():
                            summary = tf.Summary(value=[    
                                tf.Summary.Value(tag=name, simple_value=value),   
                                ])  
                            writer.add_summary(summary, global_step=step)
                    add_summary(summary_writers['train'], train_metrics, step) 
                    add_summary(summary_writers['dev'], dev_metrics, step) 
                    add_summary(summary_writers['test'], test_metrics, step)
                    trainModel.saver.save(sess, config.model_path)

            else:
                print "[MR, MRR, hits10, hits5, hits1]=", evaluate(sess, trainModel, config.test_path)

if __name__ == "__main__":
    tf.app.run()
