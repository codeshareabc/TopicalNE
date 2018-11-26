# -*- coding: utf-8 -*-

# Topical Network Embedding model.

import tensorflow as tf
import math
import numpy as np
import URLs as URL
import os
import tneutil

class TNE:
    
    def __init__(self, dataset, T, num_skip, word_window_size, node_window_size, learning_rate, topic_word_batch_size,
                  node_node_batch_size, topic_node_batch_size, topic_size,node_size, vocabulary_size,topicN,experiment_count,
                  word_embsize, topic_embsize, node_embsize, concat=True, loss_type = 'nce_loss',
                  optimize = 'Adagrad', num_sampled=5, alpha=0.4, num_runs=100000):
        self.num_skip = num_skip
        self.word_window_size = word_window_size
        self.node_window_size = node_window_size
        self.learning_rate = learning_rate
        self.topic_word_batch_size = topic_word_batch_size
        self.node_node_batch_size = node_node_batch_size
        self.topic_node_batch_size = topic_node_batch_size
        self.topic_size = topic_size
        self.word_embsize =word_embsize
        self.topic_embsize =topic_embsize
        self.node_size = node_size
        self.node_embsize =node_embsize
        self.vocabulary_size = vocabulary_size
        self.concat = concat
        self.loss_type = loss_type
        self.num_sampled = num_sampled
        self.alpha = alpha
        self.optimize = optimize
        self.num_runs = num_runs
        self.dataset = dataset
        self.topicN = topicN
        self.experiment_count = experiment_count
        self.T = T
    
    def train(self):
        
        graph = tf.Graph()
        
        with graph.as_default():
            # Input data
            topic_word_train_inputs = tf.placeholder(tf.int32, shape=[self.topic_word_batch_size, self.word_window_size+1])
            topic_word_train_labels = tf.placeholder(tf.int32, shape=[self.topic_word_batch_size, 1])
        
            node_node_train_inputs = tf.placeholder(tf.int32, shape=[self.node_node_batch_size])
            node_node_train_labels = tf.placeholder(tf.int32, shape=[self.node_node_batch_size, 1])
        
            topic_node_train_inputs = tf.placeholder(tf.float32, shape=[self.topic_node_batch_size, self.topic_size, 1]) # none of multiple shared topics between two nodes
            topic_node_train_labels = tf.placeholder(tf.int32, shape=[self.topic_node_batch_size, 1])
        
            # embeddings for words, nodes and topics
            word_embeddings = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.word_embsize],
                            stddev=1.0 / math.sqrt(self.word_embsize)))
        
            node_embeddings = tf.Variable(tf.truncated_normal([self.node_size, self.node_embsize],
                            stddev=1.0 / math.sqrt(self.node_size)))
        
            topic_embeddings = tf.Variable(tf.truncated_normal([self.topic_size, self.topic_embsize],
                            stddev=1.0 / math.sqrt(self.topic_size)))
        
            if self.concat: # concatenating word vectors and topic vector
                combined_embsize = self.word_embsize * self.word_window_size + self.topic_embsize
            else: # concatenating the average word vectors and topic vector
                combined_embsize = self.word_embsize + self.topic_embsize
        
            # define the weights and biases for the node embedding and topic embedding, respectively
            topic_word_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, combined_embsize],
                        stddev=1.0 / math.sqrt(combined_embsize)))
            topic_word_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
        
            node_node_weights = tf.Variable(
                    tf.truncated_normal([self.node_size, self.node_embsize], # all dynamically chosen topic vectors are averaged
                        stddev=1.0 / math.sqrt(self.node_embsize)))
            node_node_biases = tf.Variable(tf.zeros([self.node_size]))
        
            topic_node_weights = tf.Variable(
                    tf.truncated_normal([self.node_size, self.topic_embsize],
                        stddev=1.0 / math.sqrt(self.topic_embsize)))
            topic_node_biases = tf.Variable(tf.zeros([self.node_size]))
        
            # look up embeddings for the inputs
        
            # input embedding for topic-word training
            context_embeds = []
            if self.concat:
                for i in range(self.word_window_size):
                    word_embed = tf.nn.embedding_lookup(word_embeddings, topic_word_train_inputs[:, i])
                    context_embeds.append(word_embed)
            else: # average the vectors
                word_embed = tf.zeros([self.topic_word_batch_size, self.word_embsize])
                for i in range(self.word_window_size):
                    word_embed += tf.nn.embedding_lookup(word_embeddings, topic_word_train_inputs[:, i])
                context_embeds.append(word_embed)
            topic_embed = tf.nn.embedding_lookup(topic_embeddings, topic_word_train_inputs[:, self.word_window_size])
            context_embeds.append(topic_embed)
            topic_word_embeds = tf.concat(context_embeds, 1) # the final combined topic-word input embeddings in the PV-DM model
        
            # input embedding for the node-node training
            node_node_embeds = tf.nn.embedding_lookup(node_embeddings, node_node_train_inputs)
        
            # input embedding for the topic-node training
        #             topic_node_embeds = tf.nn.embedding_lookup(topic_embeddings, topic_node_train_inputs)
            # dynamically aggregate all shared topics, using their average embeddings
            topic_node_embeds = tf.divide(tf.reduce_sum(tf.multiply(topic_embeddings, topic_node_train_inputs[:,:]),1),
                                       tf.reduce_sum(topic_node_train_inputs[:,:],1))
        
            # compute the loss with negative sampling
            if self.loss_type == 'sampled_softmax_loss':
                topic_word_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                            weights=topic_word_weights,
                            biases=topic_word_biases,
                            labels=topic_word_train_labels,
                            inputs =topic_word_embeds,
                            num_sampled=self.num_sampled,
                            num_classes=self.vocabulary_size))
        
                node_node_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                                weights=node_node_weights,
                                biases=node_node_biases,
                                labels=node_node_train_labels,
                                inputs =node_node_embeds,
                                num_sampled=self.num_sampled,
                                num_classes=self.node_size))
        
                topic_node_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                                weights=topic_node_weights,
                                biases=topic_node_biases,
                                labels=topic_node_train_labels,
                                inputs =topic_node_embeds,
                                num_sampled=self.num_sampleds,
                                num_classes=self.node_size))
            elif self.loss_type == 'nce_loss':
                topic_word_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                            weights=topic_word_weights,
                            biases=topic_word_biases,
                            labels=topic_word_train_labels,
                            inputs =topic_word_embeds,
                            num_sampled=self.num_sampled,
                            num_classes=self.vocabulary_size))
        
                node_node_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                                weights=node_node_weights,
                                biases=node_node_biases,
                                labels=node_node_train_labels,
                                inputs =node_node_embeds,
                                num_sampled=self.num_sampled,
                                num_classes=self.node_size))
        
                topic_node_loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                                weights=topic_node_weights,
                                biases=topic_node_biases,
                                labels=topic_node_train_labels,
                                inputs =topic_node_embeds,
                                num_sampled=self.num_sampled,
                                num_classes=self.node_size))
        
            # the collective loss combing above three aspects of training loss
            global_loss = (1 - self.alpha) * node_node_loss + self.alpha * topic_node_loss + self.alpha * topic_word_loss
        
            # Optimizer.
            if self.optimize == 'Adagrad':
#                 global_step = tf.Variable(1, name="global_step", trainable=False)
#                 optimizer1 = tf.train.AdamOptimizer(self.learning_rate)
#                 grads_and_vars = optimizer1.compute_gradients(global_loss)
#                 grads, _ = list(zip(*grads_and_vars))
#                 tf.summary.scalar("gradient_norm", tf.global_norm(grads))
#                 optimizer = optimizer1.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step,
#                                                           name="train_op")
                global_step = tf.Variable(1, name="global_step", trainable=False)
                optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(global_loss, global_step=global_step)
#                 optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(global_loss)
            elif self.optimize == 'SGD':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(global_loss)
        
            # normalize the embeddings
        
            norm_node_embeddings = node_embeddings / tf.sqrt(tf.reduce_sum(tf.square(node_embeddings), 1, keep_dims=True))
            norm_topic_embeddings = topic_embeddings / tf.sqrt(tf.reduce_sum(tf.square(topic_embeddings), 1, keep_dims=True)) 
            norm_word_embeddings = word_embeddings / tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keep_dims=True)) 
        
            # Add variable initializer
            init = tf.global_variables_initializer()
            
        #     config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
        #                             allow_soft_placement=True, log_device_placement=True, device_count = {'CPU': 8})
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            with tf.Session(graph=graph,config=config) as sess:        
                init.run()
                print("Initialized!")
        
                average_loss = 0
                tw_training_batches, tw_reverse_word_dict, tw_reverse_topic_dict = tneutil.generate_batach_pvdm(self.dataset, self.T, self.topic_word_batch_size, 
                                                                                                                  self.word_window_size, topicN=self.topicN) 
                nn_training_batches, tn_training_batches, nn_reverse_node_dict, nodetopics = tneutil.generate_batch_skipgram(self.dataset, self.T, self.node_node_batch_size, 
                                                                                                                   self.topic_node_batch_size, self.node_window_size, 
                                                                                                                   self.num_skip, topicN=self.topicN)
        
                num_runs = self.num_runs
                num_out = 10000 # how many to output the loss
                batch_count = 0
                for i in range(num_runs):
                    tw_batch = tw_training_batches[i%len(tw_training_batches)]
                    nn_batch = nn_training_batches[i%len(nn_training_batches)]
                    tn_batch = tn_training_batches[i%len(tn_training_batches)]
        
                    tw_batch_inputs = np.array(tw_batch[0])
                    tw_batch_lables = np.expand_dims(np.array(tw_batch[1]), axis =1)
                    
        #             shuffle_indices = np.random.permutation(np.arange(len(tw_batch_inputs)))
        #             tw_batch_inputs = tw_batch_inputs[shuffle_indices]
        #             tw_batch_lables = tw_batch_lables[shuffle_indices]
        
                    nn_batch_inputs = np.array(nn_batch[0])
                    nn_batch_lables = np.expand_dims(np.array(nn_batch[1]), axis =1)
                    
        #             shuffle_indices = np.random.permutation(np.arange(len(nn_batch_inputs)))
        #             nn_batch_inputs = nn_batch_inputs[shuffle_indices]
        #             nn_batch_lables = nn_batch_lables[shuffle_indices]
                    
        
                    tn_batch_inputs = np.array(tn_batch[0])
                    tn_batch_lables = np.expand_dims(np.array(tn_batch[1]), axis =1)
                    
        #             shuffle_indices = np.random.permutation(np.arange(len(tn_batch_inputs)))
        #             tn_batch_inputs = tn_batch_inputs[shuffle_indices]
        #             tn_batch_lables = tn_batch_lables[shuffle_indices]
        
                    feed_dicts = {topic_word_train_inputs: tw_batch_inputs, topic_word_train_labels: tw_batch_lables,
                                  node_node_train_inputs: nn_batch_inputs, node_node_train_labels: nn_batch_lables,
                                  topic_node_train_inputs: tn_batch_inputs, topic_node_train_labels: tn_batch_lables}
        
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
                norm_topic_embeddings = sess.run(norm_topic_embeddings)
                norm_word_embeddings = sess.run(norm_word_embeddings)  
                
        node_embed_file = os.path.join(self.dataset, str(self.experiment_count)+'_'+str(self.T)+'node_embeddings.txt')
        topic_embed_file = os.path.join(self.dataset, str(self.experiment_count)+'_'+str(self.T)+'topic_embeddings.txt')
        word_embed_file = os.path.join(self.dataset, str(self.experiment_count)+'_'+str(self.T)+'word_embeddings.txt')
        nodetopic_embed_file = os.path.join(self.dataset, str(self.experiment_count)+'_'+str(self.T)+'nodetopic_embeddings.txt')
        
        
        with open(node_embed_file, 'w') as nw:
            line = str(norm_node_embeddings.shape[0]) + " " + str(norm_node_embeddings.shape[1])
            nw.write(line+'\n')
            node_dict_id = 0
            for node_embed in norm_node_embeddings:
                id = nn_reverse_node_dict[node_dict_id]
                line = str(id)
                for e in node_embed:
                    line += ' ' + str(e)
                nw.write(line + '\n')
                node_dict_id += 1
        print('Node embedding saved, %d nodes in total.' % node_dict_id)
        
        with open(topic_embed_file, 'w') as tw:
            line = str(norm_topic_embeddings.shape[0]) + " " + str(norm_topic_embeddings.shape[1])
            tw.write(line+'\n')
            topic_dict_id = 0
            for topic_embed in norm_topic_embeddings:
                id = tw_reverse_topic_dict[topic_dict_id]
                line = str(id)
                for e in topic_embed:
                    line += ' ' + str(e)
                tw.write(line + '\n')
                topic_dict_id += 1
        print('Topic embedding saved, %d topics in total.' % topic_dict_id)
        
        with open(word_embed_file, 'w', encoding='iso-8859-1') as ww:
            line = str(norm_word_embeddings.shape[0]) + " " + str(norm_word_embeddings.shape[1])
            ww.write(line+'\n')
            word_dict_id = 0
            for word_embed in norm_word_embeddings:
                id3 = tw_reverse_word_dict[word_dict_id]
                line = str(id3)
                for e in word_embed:
                    line += ' ' + str(e)
                ww.write(line + '\n')
                word_dict_id += 1
        print('Word embedding saved, %d words in total.' % word_dict_id)    
        
        # combine/concatenate the node with topic to generate the final network embedding vectors
        with open(nodetopic_embed_file, 'w') as ntw:
            embedsize = norm_node_embeddings.shape[1]+norm_topic_embeddings.shape[1]
            line = str(norm_node_embeddings.shape[0]) + " " + str(embedsize)
            ntw.write(line + '\n')
            nodetopic_dict_id = 0
            for node_vec in norm_node_embeddings:
                node_name = nn_reverse_node_dict[nodetopic_dict_id]
                topic_vec_sum = np.zeros([self.topic_embsize,])
                for topic in nodetopics[nodetopic_dict_id]:
                    topic_vec_sum += norm_topic_embeddings[topic]
                if len(nodetopics[nodetopic_dict_id]) != 0:
                    topic_vec_sum = topic_vec_sum / len(nodetopics[nodetopic_dict_id])
                    line = str(node_name)
                    for e in node_vec:
                        line += ' ' + str(e)
                    for e in topic_vec_sum:
                        line += ' ' + str(e)
                    ntw.write(line + '\n')
                nodetopic_dict_id += 1
        print('Nodetopic embedding saved, %d words in total.' % nodetopic_dict_id)           
                       
def evaluate_nodevec(dataset, T, experiment_count, train_size, experiment_num):
    from sklearn.svm import LinearSVC
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    
    from sklearn.cross_validation import train_test_split
    
    # prepare the data for node classification
    all_data = dict()
    labelsfile = os.path.join(dataset, "labels.txt")
    with open(labelsfile) as lr:
        for line in lr:
            params = line.split()
            all_data[params[0]] = params[1]
    
    
#     print(len(all_data))
#     print(len(train))
    
    allvecs = dict()
    # node_embed_file = os.path.join(dataset, 'node_embeddings.txt')
#     node_embed_file = os.path.join(dataset, str(experiment_count)+'_'+str(T)+'node_embeddings.txt')
    node_embed_file = os.path.join(dataset, str(experiment_count)+'node_embeddings.txt')
    with open(node_embed_file) as er:
        er.readline()
        for line in er:
            params = line.split()
            node_id = params[0]
            node_embeds = [float(value) for value in params[1:]]
            allvecs[node_id] = node_embeds
    
    macro_f1s = []
    micro_f1s = []
    for experiment_time in range(experiment_num):
        train, test = train_test_split(list(all_data.keys()), train_size=train_size, random_state=experiment_time)
        train_vec = []
        train_y = []
        test_vec = []
        test_y = []
        for train_i in train:
            train_vec.append(allvecs[train_i])
            train_y.append(all_data[train_i])
        for test_i in test:
            test_vec.append(allvecs[test_i])
            test_y.append(all_data[test_i])
        
        classifier = LinearSVC()
        classifier.fit(train_vec, train_y)
        y_pred = classifier.predict(test_vec)
        
        cm = confusion_matrix(test_y, y_pred)
#         print(cm)
        
        acc = accuracy_score(test_y, y_pred)
        macro_recall = recall_score(test_y, y_pred,  average='macro')
        macro_f1 = f1_score(test_y, y_pred,pos_label=None, average='macro')
        micro_f1 = f1_score(test_y, y_pred,pos_label=None, average='micro')
        
        macro_f1s.append(macro_f1)
        micro_f1s.append(micro_f1)
        
        print('experiment_time: %d, Classification Accuracy=%f, macro_recall=%f macro_f1=%f, micro_f1=%f' % (experiment_time, acc, macro_recall, macro_f1, micro_f1))              
        
    import statistics
    average_macro_f1 = statistics.mean(macro_f1s)
    average_micro_f1 = statistics.mean(micro_f1s)
    stdev_macro_f1 = statistics.stdev(macro_f1s)
    stdev_micro_f1 = statistics.stdev(micro_f1s)
    
    print('Total experiment time: %d, average_macro_f1=%f, stdev_macro_f1=%f, average_micro_f1=%f, stdev_micro_f1=%f' % (experiment_num, average_macro_f1, stdev_macro_f1, average_micro_f1, stdev_micro_f1))

                    
def evaluate_nodetopicvec(dataset, T, experiment_count, train_size, experiment_num):
    from sklearn.svm import LinearSVC
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score
    
    from sklearn.cross_validation import train_test_split
    
    # prepare the data for node classification
    all_data = dict()
    nt_labelsfile = os.path.join(dataset, "labels.txt")
    with open(nt_labelsfile) as lr:
        for line in lr:
            params = line.split()
            for lab in params[1].split(','):
                all_data[params[0]] = lab
    
    
#     print(len(all_data))
#     print(len(train))
    
    allvecs = dict()
    # nodetopic_embed_file = os.path.join(dataset, 'nodetopic_embeddings.txt')
    nodetopic_embed_file = os.path.join(dataset, str(experiment_count)+'nodetopic_embeddings.txt')
#     nodetopic_embed_file = os.path.join(dataset, str(experiment_count)+'_'+str(T)+'nodetopic_embeddings.txt')
    with open(nodetopic_embed_file) as er:
        er.readline()
        for line in er:
            params = line.split()
            node_id = params[0]
            node_embeds = [float(value) for value in params[1:]]
            allvecs[node_id] = node_embeds
    
    
    
    macro_f1s = []
    micro_f1s = []
    for experiment_time in range(experiment_num):
        train, test = train_test_split(list(all_data.keys()), train_size=train_size, random_state=experiment_time)
        train_vec = []
        train_y = []
        test_vec = []
        test_y = []
        for train_i in train:
            train_vec.append(allvecs[train_i])
            train_y.append(all_data[train_i])
        for test_i in test:
            test_vec.append(allvecs[test_i])
            test_y.append(all_data[test_i])
        
        classifier = LinearSVC()
        classifier.fit(train_vec, train_y)
        y_pred = classifier.predict(test_vec)
        
        cm = confusion_matrix(test_y, y_pred)
#         print(cm)
        
        acc = accuracy_score(test_y, y_pred)
        macro_recall = recall_score(test_y, y_pred,  average='macro')
        macro_f1 = f1_score(test_y, y_pred,pos_label=None, average='macro')
        micro_f1 = f1_score(test_y, y_pred,pos_label=None, average='micro')
        
        macro_f1s.append(macro_f1)
        micro_f1s.append(micro_f1)
        
        print('experiment_time: %d, Classification Accuracy=%f, macro_recall=%f macro_f1=%f, micro_f1=%f' % (experiment_time, acc, macro_recall, macro_f1, micro_f1))              
        
    import statistics
    average_macro_f1 = statistics.mean(macro_f1s)
    average_micro_f1 = statistics.mean(micro_f1s)
    stdev_macro_f1 = statistics.stdev(macro_f1s)
    stdev_micro_f1 = statistics.stdev(micro_f1s)
    
    print('Total experiment time: %d, average_macro_f1=%f, stdev_macro_f1=%f, average_micro_f1=%f, stdev_micro_f1=%f' % (experiment_num, average_macro_f1, stdev_macro_f1, average_micro_f1, stdev_micro_f1))

def main():
    dataset = URL.FILES.citeseerData
    num_skip = 4
    word_window_size = 2
    node_window_size = 4
    learning_rate = 0.05
    topic_word_batch_size = 120
    node_node_batch_size = 120
    topic_node_batch_size = 120
    word_embsize = 100
    topic_embsize = 100
    node_embsize = 100
    num_sampled = 5
    num_runs = 200000
#     num_runs = 50000
    loss_type = 'nce_loss'
    optimize = 'Adagrad'
    concat = False 
    alpha = 0.2
#     experiment_count = 2 
    
    topicN=2
    T = 20
    
    node_size, topic_size, vocabulary_size = tneutil.repository_size(dataset, T, topicN)
    print('node_size:{}, label_size:{}, word_size:{}, T:{}'.format(node_size,topic_size,vocabulary_size, T))
    
    '''
    num_skip = 4, word_window_size = 2, node_window_size = 4, learning_rate = 0.05, 
    topic_word_batch_size = 120, node_node_batch_size = 120, topic_node_batch_size = 120, 
    word_embsize = 100, topic_embsize = 100, node_embsize = 100, num_sampled = 5, alpha = 0.2
    num_runs = 200000, topicN=1
    '''
#     experiment_count = 1 
    '''
    num_skip = 4, word_window_size = 2, node_window_size = 4, learning_rate = 0.05, 
    topic_word_batch_size = 120, node_node_batch_size = 120, topic_node_batch_size = 120, 
    word_embsize = 100, topic_embsize = 100, node_embsize = 100, num_sampled = 5, alpha = 0.2
    num_runs = 200000, topicN=2
    '''
    experiment_count = 2
    '''
    num_skip = 4, word_window_size = 2, node_window_size = 4, learning_rate = 0.05, 
    topic_word_batch_size = 120, node_node_batch_size = 120, topic_node_batch_size = 120, 
    word_embsize = 100, topic_embsize = 100, node_embsize = 100, num_sampled = 5, alpha = 0.2
    num_runs = 200000, topicN=3
    '''
#     experiment_count = 3
    '''
    num_skip = 4, word_window_size = 2, node_window_size = 4, learning_rate = 0.05, 
    topic_word_batch_size = 120, node_node_batch_size = 120, topic_node_batch_size = 120, 
    word_embsize = 100, topic_embsize = 100, node_embsize = 100, num_sampled = 5, alpha = 0.2
    num_runs = 200000, topicN=4
    '''
#     experiment_count = 4
    '''
    num_skip = 4, word_window_size = 2, node_window_size = 4, learning_rate = 0.05, 
    topic_word_batch_size = 120, node_node_batch_size = 120, topic_node_batch_size = 120, 
    word_embsize = 100, topic_embsize = 100, node_embsize = 100, num_sampled = 5, alpha = 0.2
    num_runs = 200000, topicN=5
    '''
#     experiment_count = 5
    '''
    num_skip = 4, word_window_size = 2, node_window_size = 4, learning_rate = 0.05, 
    topic_word_batch_size = 120, node_node_batch_size = 120, topic_node_batch_size = 120, 
    word_embsize = 100, topic_embsize = 100, node_embsize = 100, num_sampled = 5, alpha = 0.2
    num_runs = 200000, topicN=6
    '''
#     experiment_count = 6
    
    # test the influence of alpha topicN=2
#     experiment_count = 'alpha1_'
#     alpha = 0.1
    
#     experiment_count = 'alpha2_'
#     alpha = 0.3
#     
#     experiment_count = 'alpha3_'
#     alpha = 0.4
#     
#     experiment_count = 'alpha4_'
#     alpha = 0.5
#     
#     experiment_count = 'alpha5_'
#     alpha = 0.6
#     
#     experiment_count = 'alpha6_'
#     alpha = 0.7
#     
#     experiment_count = 'alpha7_'
#     alpha = 0.8
    
    # test of node dimension topicN=2, alpha = 0.2
#     experiment_count = 'nodedim1_'
#     node_embsize = 50
    
#     experiment_count = 'nodedim2_'
#     node_embsize = 150
#     
#     experiment_count = 'nodedim3_'
#     node_embsize = 200
#     
#     experiment_count = 'nodedim4_'
#     node_embsize = 250
#     
#     experiment_count = 'nodedim5_'
#     node_embsize = 300
#     
#     experiment_count = 'nodedim6_'
#     node_embsize = 350
#     
#     experiment_count = 'nodedim7_'
#     node_embsize = 400
    
    experiment_num = 20 # number of evaluation times for each train size, the average result and standard deviation are calculated    
    train_size = 0.1
    
#     tne = TNE(dataset=dataset,T=T,
#               num_skip=num_skip, word_window_size=word_window_size, 
#               node_window_size=node_window_size, learning_rate=learning_rate, 
#               topic_word_batch_size=topic_word_batch_size, node_node_batch_size=node_node_batch_size,
#               topic_node_batch_size=topic_node_batch_size, topic_size=topic_size,
#               node_size=node_size, vocabulary_size=vocabulary_size,topicN=topicN,experiment_count=experiment_count,
#               word_embsize=word_embsize, topic_embsize=topic_embsize, 
#               node_embsize=node_embsize,
#               num_sampled=num_sampled,
#               alpha=alpha,num_runs=num_runs,
#               concat=concat,  
#               loss_type=loss_type,
#               optimize=optimize)    
#           
#     tne.train()
    print('experiment_count:{}, train_size:{}'.format(experiment_count, train_size))
    evaluate_nodevec(dataset, T, experiment_count, train_size, experiment_num)
    evaluate_nodetopicvec(dataset, T, experiment_count, train_size, experiment_num)
if __name__ == '__main__':
    main() 
                