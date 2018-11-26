# -*- coding: utf-8 -*-

# TriDNR Network Embedding model.

import tensorflow as tf
import math
import numpy as np
import URLs as URL
import os
import tridnrutil as modelutil

class TriDNR:
    
    def __init__(self, dataset, experiment_count, num_skip, word_window_size, node_window_size, learning_rate, node_word_batch_size,
                  node_node_batch_size, group_word_batch_size, group_size,node_size, vocabulary_size,
                  word_embsize, group_embsize, node_embsize, train_data, loss_type = 'nce_loss',
                  optimize = 'Adagrad', num_sampled=5, alpha=0.4, num_runs=100000):
        
        self.num_skip = num_skip
        self.word_window_size = word_window_size
        self.node_window_size = node_window_size
        self.learning_rate = learning_rate
        self.node_word_batch_size = node_word_batch_size
        self.node_node_batch_size = node_node_batch_size
        self.group_word_batch_size = group_word_batch_size
        self.group_size = group_size
        self.word_embsize =word_embsize
        self.group_embsize =group_embsize
        self.node_size = node_size
        self.node_embsize =node_embsize
        self.vocabulary_size = vocabulary_size
        self.loss_type = loss_type
        self.num_sampled = num_sampled
        self.alpha = alpha
        self.optimize = optimize
        self.num_runs = num_runs
        self.dataset = dataset
        self.train_data = train_data
        self.experiment_count = experiment_count

    def train(self):
        
        graph = tf.Graph()
        
        with graph.as_default():
            # Input data
            node_word_train_inputs = tf.placeholder(tf.int32, shape=[self.node_word_batch_size])
            node_word_train_labels = tf.placeholder(tf.int32, shape=[self.node_word_batch_size, 1])
            
            group_word_train_inputs = tf.placeholder(tf.int32, shape=[self.group_word_batch_size]) 
            group_word_train_labels = tf.placeholder(tf.int32, shape=[self.group_word_batch_size, 1])
            
            node_node_train_inputs = tf.placeholder(tf.int32, shape=[self.node_node_batch_size])
            node_node_train_labels = tf.placeholder(tf.int32, shape=[self.node_node_batch_size, 1])
            
            # embeddings for words, nodes and topics
            word_embeddings = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.word_embsize],
                            stddev=1.0 / math.sqrt(self.vocabulary_size)))
            
            node_embeddings = tf.Variable(tf.truncated_normal([self.node_size, self.node_embsize],
                            stddev=1.0 / math.sqrt(self.node_size)))
            
            group_embeddings = tf.Variable(tf.truncated_normal([self.group_size, self.group_embsize],
                            stddev=1.0 / math.sqrt(self.group_size)))
            
            
            # define the weights and biases for the node embedding and topic embedding, respectively
            node_word_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.node_embsize],
                        stddev=1.0 / math.sqrt(self.node_embsize)))
            node_word_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
            
            group_word_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.group_embsize],
                        stddev=1.0 / math.sqrt(self.group_embsize)))
            group_word_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
            
            node_node_weights = tf.Variable(tf.truncated_normal([self.node_size, self.node_embsize],
                        stddev=1.0 / math.sqrt(self.node_embsize)))
            node_node_biases = tf.Variable(tf.zeros([self.node_size]))
            
            # look up embeddings for the inputs
            # input embedding for the node-node training
            node_node_embeds = tf.nn.embedding_lookup(node_embeddings, node_node_train_inputs)
            # input embedding for the node-word training
            node_word_embeds = tf.nn.embedding_lookup(node_embeddings, node_word_train_inputs)
            # input embedding for the group-word training
            group_word_embeds = tf.nn.embedding_lookup(group_embeddings, group_word_train_inputs)
            
            # compute the loss with negative sampling
            if self.loss_type == 'sampled_softmax_loss':
                node_word_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                            weights=node_word_weights,
                            biases=node_word_biases,
                            labels=node_word_train_labels,
                            inputs =node_word_embeds,
                            num_sampled=self.num_sampled,
                            num_classes=self.vocabulary_size))
                
                group_word_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                                weights=group_word_weights,
                                biases=group_word_biases,
                                labels=group_word_train_labels,
                                inputs =group_word_embeds,
                                num_sampled=self.num_sampleds,
                                num_classes=self.vocabulary_size))
                
                node_node_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                                weights=node_node_weights,
                                biases=node_node_biases,
                                labels=node_node_train_labels,
                                inputs =node_node_embeds,
                                num_sampled=self.num_sampled,
                                num_classes=self.node_size))
            
            elif self.loss_type == 'nce_loss':
                node_word_loss = tf.reduce_mean(tf.nn.nce_loss(
                            weights=node_word_weights,
                            biases=node_word_biases,
                            labels=node_word_train_labels,
                            inputs =node_word_embeds,
                            num_sampled=self.num_sampled,
                            num_classes=self.vocabulary_size))
                
                group_word_loss = tf.reduce_mean(tf.nn.nce_loss(
                                weights=group_word_weights,
                                biases=group_word_biases,
                                labels=group_word_train_labels,
                                inputs =group_word_embeds,
                                num_sampled=self.num_sampled,
                                num_classes=self.vocabulary_size))
                
                node_node_loss = tf.reduce_mean(tf.nn.nce_loss(
                                weights=node_node_weights,
                                biases=node_node_biases,
                                labels=node_node_train_labels,
                                inputs =node_node_embeds,
                                num_sampled=self.num_sampled,
                                num_classes=self.node_size))
                
            # the collective loss combing above three aspects of training loss
            global_loss = (1 - self.alpha) * node_node_loss + self.alpha * node_word_loss + self.alpha * group_word_loss
            
            # Optimizer.
            if self.optimize == 'Adagrad':
                global_step = tf.Variable(1, name="global_step", trainable=False)
                optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(global_loss, global_step=global_step)
            elif self.optimize == 'SGD':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(global_loss)

            # normalize the embeddings
            norm_node_embeddings = node_embeddings / tf.sqrt(tf.reduce_sum(tf.square(node_embeddings), 1, keep_dims=True))
            norm_group_embeddings = group_embeddings / tf.sqrt(tf.reduce_sum(tf.square(group_embeddings), 1, keep_dims=True)) 
            norm_word_embeddings = word_embeddings / tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keep_dims=True)) 

            # Add variable initializer
            init = tf.global_variables_initializer()
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            with tf.Session(graph=graph,config=config) as sess:
                init.run()
                print("Initialized!")
                
                average_loss = 0
                
                nn_training_batches, reverse_node_dict= modelutil.generate_batch_deepwalk(self.dataset, self.node_node_batch_size, 
                                                               self.num_skip, self.node_window_size)
                nw_training_batches, gw_training_batches, reverse_group_dict, reverse_word_dict = modelutil.generate_batch_triDNR(self.dataset, self.node_word_batch_size, 
                                                                                                                                  self.group_word_batch_size, self.train_data)
                
                num_runs = self.num_runs
                num_out = 10000 # how many to output the loss
                batch_count = 0
                for i in range(num_runs):
                    nn_batch = nn_training_batches[i%len(nn_training_batches)]
                    nw_batch = nw_training_batches[i%len(nw_training_batches)]
                    gw_batch = gw_training_batches[i%len(gw_training_batches)]  
                    
                    nn_batch_inputs = np.array(nn_batch[0])
                    nn_batch_lables = np.expand_dims(np.array(nn_batch[1]), axis =1)
                    
                    nw_batch_inputs = np.array(nw_batch[0])
                    nw_batch_lables = np.expand_dims(np.array(nw_batch[1]), axis =1)
                    
                    gw_batch_inputs = np.array(gw_batch[0])
                    gw_batch_lables = np.expand_dims(np.array(gw_batch[1]), axis =1)
                                      
                    feed_dicts = {node_node_train_inputs: nn_batch_inputs, node_node_train_labels: nn_batch_lables,
                                  node_word_train_inputs: nw_batch_inputs, node_word_train_labels: nw_batch_lables,
                                  group_word_train_inputs: gw_batch_inputs, group_word_train_labels: gw_batch_lables}
 
                    # run the graph
                    sess.run(optimizer, feed_dict=feed_dicts)
                    loss_val = sess.run(global_loss, feed_dict=feed_dicts)
                    average_loss += loss_val
                    
                    batch_count += 1
                    
                    if i % num_out == 0:
                        average_loss = average_loss / batch_count
                        print("num runs = {}, average loss = {}".format(i, average_loss))
                
                # Save embeddings to local disk
                norm_node_embeddings = sess.run(norm_node_embeddings)
                norm_group_embeddings = sess.run(norm_group_embeddings)
                norm_word_embeddings = sess.run(norm_word_embeddings)
                
        node_embed_file = os.path.join(self.dataset, 'triDNR', str(self.experiment_count)+'node_embeddings.txt')
        topic_embed_file = os.path.join(self.dataset, 'triDNR', str(self.experiment_count)+'group_embeddings.txt')
        word_embed_file = os.path.join(self.dataset, 'triDNR', str(self.experiment_count)+'word_embeddings.txt')
        
        with open(node_embed_file, 'w') as nw:
            line = str(norm_node_embeddings.shape[0]) + " " + str(norm_node_embeddings.shape[1])
            nw.write(line+'\n')
            node_dict_id = 0
            for node_embed in norm_node_embeddings:
                id1 = reverse_node_dict[node_dict_id]
                line = str(id1)
                for e in node_embed:
                    line += ' ' + str(e)
                nw.write(line + '\n')
                node_dict_id += 1
        print('Node embedding saved, %d nodes in total.' % node_dict_id)
        
        with open(topic_embed_file, 'w') as tw:
            line = str(norm_group_embeddings.shape[0]) + " " + str(norm_group_embeddings.shape[1])
            tw.write(line+'\n')
            topic_dict_id = 0
            for topic_embed in norm_group_embeddings:
                id2 = reverse_group_dict[topic_dict_id]
                line = str(id2)
                for e in topic_embed:
                    line += ' ' + str(e)
                tw.write(line + '\n')
                topic_dict_id += 1
        print('Group embedding saved, %d topics in total.' % topic_dict_id)
        
        with open(word_embed_file, 'w', encoding='iso-8859-1') as ww:
            line = str(norm_word_embeddings.shape[0]) + " " + str(norm_word_embeddings.shape[1])
            ww.write(line+'\n')
            word_dict_id = 0
            for word_embed in norm_word_embeddings:
                id3 = reverse_word_dict[word_dict_id]
                line = str(id3)
                for e in word_embed:
                    line += ' ' + str(e)
                ww.write(line + '\n')
                word_dict_id += 1
        print('Word embedding saved, %d words in total.' % word_dict_id)    
                
def evaluate(dataset, experiment_count, train_size):
    from sklearn.svm import LinearSVC
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    
    train_indexfile = os.path.join(dataset, 'triDNR', str(experiment_count)+'train_index.txt')
    test_indexfile = os.path.join(dataset, 'triDNR', str(experiment_count)+'test_index.txt')
    
    with open(train_indexfile, 'r') as rtrain, open(test_indexfile, 'r') as rtest:
        line = rtrain.readline()
        train = line.split()
        line = rtest.readline()
        test = line.split() 
    
    # prepare the data for node classification
    all_data = dict()
    labelsfile = os.path.join(dataset, "labels.txt")
    with open(labelsfile) as lr:
        for line in lr:
            params = line.split()
            for lab in params[1].split(','):
                all_data[params[0]] = lab
    
    
    print(len(all_data))
    print(len(train))
    
    allvecs = dict()
    node_embed_file = os.path.join(dataset, 'triDNR', str(experiment_count)+'node_embeddings.txt')
    with open(node_embed_file) as er:
        er.readline()
        for line in er:
            params = line.split()
            node_id = params[0]
            node_embeds = [float(value) for value in params[1:]]
            allvecs[node_id] = node_embeds
    
    train_vec = []
    train_y = []
    test_vec = []
    test_y = []
    for train_i in train:
        if not train_i in all_data.keys():
            continue
        train_vec.append(allvecs[train_i])
        train_y.append(all_data[train_i])
    for test_i in test:
        if not test_i in all_data.keys():
            continue
        test_vec.append(allvecs[test_i])
        test_y.append(all_data[test_i])
    
    classifier = LinearSVC()
    classifier.fit(train_vec, train_y)
    y_pred = classifier.predict(test_vec)
    
    cm = confusion_matrix(test_y, y_pred)
    print(cm)
    
    acc = accuracy_score(test_y, y_pred)
    macro_recall = recall_score(test_y, y_pred,  average='macro')
    macro_f1 = f1_score(test_y, y_pred,pos_label=None, average='macro')
    micro_f1 = f1_score(test_y, y_pred,pos_label=None, average='micro')
    
    print('Training data ratio=%f, classification Accuracy=%f, macro_recall=%f macro_f1=%f, micro_f1=%f' % (train_size, acc, macro_recall, macro_f1, micro_f1))
       
                
def main():
    from sklearn.cross_validation import train_test_split
    dataset = URL.FILES.citeseerData
    num_skip = 4
    word_window_size = 4
    node_window_size = 4
    learning_rate = 0.05
    node_word_batch_size = 120
    node_node_batch_size = 120
    group_word_batch_size = 120
    word_embsize = 100
    node_embsize = 100
    group_embsize = 100
    num_sampled = 5
    alpha = 0.2
    num_runs = 200000
#     num_runs = 50000
    loss_type = 'nce_loss'
    optimize = 'Adagrad'
    node_size, group_size, vocabulary_size = modelutil.repository_size(dataset)  
    print('node_size:{}, group_size:{}, word_size:{}'.format(node_size,group_size,vocabulary_size))    
    
    experiment_count = 2
    train_size = 0.7
    
    train_data, test = train_test_split(list(range(node_size)), train_size=train_size, random_state=experiment_count)
     
    train_indexfile = os.path.join(dataset, 'triDNR', str(experiment_count)+'train_index.txt')
    test_indexfile = os.path.join(dataset, 'triDNR', str(experiment_count)+'test_index.txt')
    with open(train_indexfile, 'w') as trainw, open(test_indexfile, 'w') as testw:
        line = ' '.join([str(value) for value in train_data])
        trainw.write(line)
        line = ' '.join([str(value) for value in test])
        testw.write(line)
       
#     tne = TriDNR(dataset=dataset,experiment_count=experiment_count,
#               num_skip=num_skip, word_window_size=word_window_size, 
#               node_window_size=node_window_size, learning_rate=learning_rate, 
#               node_word_batch_size=node_word_batch_size, node_node_batch_size=node_node_batch_size,
#               group_word_batch_size=group_word_batch_size, group_size=group_size,
#               node_size=node_size, vocabulary_size=vocabulary_size,
#               word_embsize=word_embsize, group_embsize=group_embsize, 
#               node_embsize=node_embsize,num_sampled=num_sampled,
#               alpha=alpha,num_runs=num_runs,loss_type=loss_type,optimize=optimize, train_data = train_data)      
#     tne.train()

    print('experiment_count:{}, train_size:{}'.format(experiment_count, train_size))
    evaluate(dataset, experiment_count, train_size)
if __name__ == '__main__':
    main() 