import tensorflow as tf
from config import Config
from data_utils import load_data
from model import TransEModel 
import sys

def read_graph(train_path):
        def dt():
            return defaultdict(set)
        nexts=defaultdict(dt)
        links=defaultdict(dt)
        before=defaultdict(dt)
        
        for line in open(train_path):
            e1,e2,rel=line.strip().split()
            nexts[e1][rel].add(e2)
            links[e1][e2].add(rel)
            before[e2][rel].add(e1)
        return nexts, links, before
            

def main(_):
    config = Config()
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer(uniform = False)
            with tf.variable_scope("model", reuse=None, initializer = initializer):
                trainModel = TransEModel(config = config)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(0.001)
            grads_and_vars = optimizer.compute_gradients(trainModel.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver()
            sess.run(tf.initialize_all_variables())
            if (config.loadFromData):
                saver.restore(sess, config.model_path)
                     
            def train_step(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
                feed_dict = {
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                    trainModel.neg_h: neg_h_batch,
                    trainModel.neg_t: neg_t_batch,
                    trainModel.neg_r: neg_r_batch
                }
                _, step, loss = sess.run(
                    [train_op, global_step, trainModel.loss], feed_dict)
                return loss

            def test_step(pos_h_batch, pos_t_batch, pos_r_batch):
                feed_dict = {
                    trainModel.pos_h: pos_h_batch,
                    trainModel.pos_t: pos_t_batch,
                    trainModel.pos_r: pos_r_batch,
                }
                step, predict = sess.run(
                    [global_step, trainModel.predict], feed_dict)
                return predict

            if sys.argv[1]=="train":
                data=load_data(config,"train")
                for times in range(config.trainTimes):
                    res = 0.0
                    for k,batch_data in enumerate(data):
                        ph, pt, pr, nh, nt, nr = batch_data
                        res += train_step(ph, pt, pr, nh, nt, nr)
                        current_step = tf.train.global_step(sess, global_step)
                        if k%1000==0:
                            print k, res/k
                    print times
                    print res
                    saver.save(sess, config.model_path)
            else:
                nexts, links=read_graph(config.train_path)
                rel_emb=sess.run(trainModel.rel_embeddings)
                ent_emb=sess.run(trainModel.ent_embeddings)
                print "ent_shape", ent_emb.shape
                print "rel_shape", rel_emb.shape
                
                
                def sum_emb(path):
                    a=rel_emb[int(path[0])]
                    for p in path[1:]:
                        a=a+rel_emb[int(p)]
                    return a
                
                def rank(v,v_,emb,name,trainset):
                    dist=np.linalg.norm(v - v_, ord=1)
                    dists = [np.linalg.norm(v_ - emb[i], ord=1) for i in range(emb.shape[0])]
                    rank=0
                    for k,dist_ in enumerate(dists):
                        if k in trainset:continue
                        if dist_<dist:
                            if rank<=10:
                                print k, name.get(k,None)
                                pass
                            rank+=1
                    return rank
                rank_tails=[]
                rank_rels=[]
                for line in open(config.test_path).readlines()[1:]:
                    try:
                            e1, e2 ,rel = line.strip().split()
                            trainset_e2=(entity2id[x] for x in nexts[e1][rel])
                            trainset_r=(relation2id[x] for x in links[e1][e2])
                            e1,e2,rel = entity2id[e1], entity2id[e2], relation2id[rel]
                            print e1,e2,rel
                            print eid2name[e1],eid2name[e2],rid2name[rel]
                            v1, v2, vr = ent_emb[int(e1)], ent_emb[int(e2)], rel_emb[int(rel)]
                            v2_=v1+vr
                            vr_=v2-v1
                            print '-'*20+"predict tail from <head, relation>"+'-'*20
                            rank_tail=rank(v2, v2_, ent_emb, eid2name, trainset_e2)
                            print rank_tail
                            print '-'*20+"predict relation from <head, tail>"+'-'*20
                            rank_rel=rank(vr, vr_, rel_emb, rid2name, trainset_r)
                            print rank_rel
                            raw_input("XXXXXXX")
                            rank_tails.append(rank_tail)
                            rank_rels.append(rank_rel)
                    except Exception,e:
                            print e
                            import traceback
                            traceback.print_exc()
                            pass
                    print sum(rank_tails)/len(rank_tails), sum(rank_rels)/len(rank_rels)

if __name__ == "__main__":
    tf.app.run()
