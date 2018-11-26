# -*- coding: utf-8 -*-
from _overlapped import NULL

import collections
import os
import gensim
import random
import gensim.utils as ut
import URLs as URL
import numpy as np
import gc
from collections import namedtuple
from itertools import compress
from numpy import dtype
from deepwalk import graph


WordTopic = namedtuple('WordTopic', 'words topics docId')
NodeTopic = namedtuple('NodeTopic', 'nodes topics')

def loadNetworkContent(dir, T, stemmer=0, topicN=1):
    alldocs = []
    contentfile = os.path.join(dir,"content_token.txt")
    labelfile = os.path.join(dir,'topics'+str(topicN)+'_T_'+ str(T)+'.txt')
    
    doc_labels = dict()
    with open(labelfile) as lr:
        for lline in lr:
            params = lline.split()
            doc_labels[int(params[0])] = params[1].split(",")
    
    with open(contentfile, 'r', encoding="iso-8859-1") as cr:
        linecount = 0
        for cline in cr:
            if stemmer == 1:
                cline = gensim.parsing.stem_text(cline)
            else:
                cline = cline.lower()
            words = ut.to_unicode(cline).split()
            topics = []
            if linecount in doc_labels.keys():
                topics= doc_labels[linecount]
            alldocs.append(WordTopic(words, topics, str(linecount)))
            linecount += 1
    return alldocs

# alldocs = loadNetworkContent(URL.FILES.serviceData)   
# print(alldocs[0]) 

def generateWalks(dir, T=5, number_walks=20, walk_length=10, topicN=1):
    adjlistfile = os.path.join(dir,"adjlist.txt")
    print('topicNtopicN:',topicN)
    print('number_walks:',number_walks)
    print('T:',T)
#     labelfile = os.path.join(dir,"topics"+str(topicN)+".txt")
    labelfile = os.path.join(dir,'topics'+str(topicN)+'_T_'+ str(T)+'.txt')
    G = graph.load_adjacencylist(adjlistfile)
    num_walks = len(G.nodes()) * number_walks
    print("Number of nodes:", len(G.nodes()))
    print("Number of walks:", num_walks)
    
    print("walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths= number_walks, 
                                        path_length=walk_length, alpha=0, rand=random.Random(1))
    node2topics = dict()
    unique_labels = []
    with open(labelfile) as lr:
        for lline in lr:
            node_index = lline.split()[0]
            node2topics[node_index] = lline.split()[1].split(",")
            for label in lline.split()[1].split(","):
                if not label in unique_labels:
                    unique_labels.append(label)
    
    topical_raw_walks = []
    walkfile = os.path.join(dir,"walks.txt")
    with open(walkfile, 'w') as ww:
        for i, x in enumerate(walks):
            nodes = [str(t) for t in x]
            topics = [node2topics[str(t)] for t in x]
            topical_raw_walks.append(NodeTopic(nodes, topics))
            
            ww.write(' '.join(nodes)+"\n")
    return topical_raw_walks, len(unique_labels), node2topics
    
# def main():
#     generateWalks(URL.FILES.serviceData, 50, 10)
# if __name__ == '__main__':
#     main()

def build_dictionary(raw_data, type):
    
    item_dictionary = dict()
    topic_dictionary = dict()
    all_items = []
    for data_item in raw_data:
        items = data_item[0] # words or nodes
        topics = data_item[1] # topics
        all_items.extend(items)
        
        if type == 'content':
            for topic in topics:
                topic_dictionary[topic] = int(topic[1:]) # remove the first 'L' character
        elif type == 'network':
            for ts in topics:
                for t in ts:
                    topic_dictionary[t] = int(t[1:]) # remove the first 'L' character
        else:
            raise Exception('dictionary must be assigned.')
        
    item_count = []
    item_count.extend(collections.Counter(all_items).most_common())
    for item,_ in item_count:
        item_dictionary[item] = len(item_dictionary)
    
    return item_dictionary, topic_dictionary

def generate_batach_pvdm(dir, T, topic_word_batch_size, word_window_size, topicN=1):
    '''
    Batch generator for PV-DM (Distributed Memory Model of Paragraph Vectors).
    batch should be a shape of (topic_word_batch_size, word_window_size+1), where 1 is for topic
    
    Parameters
    ------------
    dir: dataset root directory
    topic_word_batch_size: number of words in each mini-batch
    word_window_size: number of context words to infer the target word
    '''
    print("Generating the training instances for PV-DM model...")
    
    alldocs = loadNetworkContent(dir, T, 0, topicN)
    word_dict, topic_dict = build_dictionary(alldocs, 'content')
    reverse_word_dict = dict(zip(word_dict.values(), word_dict.keys()))
    reverse_topic_dict = dict(zip(topic_dict.values(), topic_dict.keys()))
    
    # skip_window = word_window_size / 2 # the number of words chosen around (left of right) the target word
    span = word_window_size + 1
#     skip_window = int(word_window_size // 2)
    mask = [1] * span
    mask[-1] = 0
    
    all_instances = []
    all_labels = []
    for wordtopic in alldocs:
        topics = wordtopic.topics
        words = wordtopic.words
        if len(words) < span:
            continue
        for topic in topics:
            data_index = 0
            buffer = collections.deque(maxlen=span)
            for _ in range(span):
                buffer.append(word_dict[words[data_index]])
                data_index += 1
              
            context = list(compress(buffer, mask)) + [topic_dict[topic]]
            target = buffer[-1]
            all_instances.append(context)
            all_labels.append(target)
              
            while data_index < len(words):
                buffer.append(word_dict[words[data_index]])
                data_index += 1
                  
                context = list(compress(buffer, mask)) + [topic_dict[topic]]
                target = buffer[-1]
                all_instances.append(context)
                all_labels.append(target)
                
#             for center_index in range(word_window_size, len(words)-(word_window_size+1)):
#                 buffer = []
#                 
#                 data_index = center_index
#                 for _ in range(skip_window):
#                     data_index = data_index - 1
#                     if data_index >= 0:
#                         buffer.append(word_dict[words[data_index]])
#                         
#                 data_index = center_index
#                 for _ in range(skip_window):
#                     data_index = data_index + 1
#                     if data_index <= (len(words) - 1):
#                         buffer.append(word_dict[words[data_index]]);
#                 context = list(compress(buffer, mask)) + [topic_dict[topic]]
#                 
#                 all_instances.append(context)
#                 all_labels.append(word_dict[words[center_index]])
            
#     for wordtopic in alldocs:
#         topics = wordtopic.topics
#         words = wordtopic.words
#         if len(words) < span:
#             continue
#         for topic in topics:
#             data_index = 0
#             buffer = collections.deque(maxlen=span)
#             for _ in range(span):
#                 buffer.append(word_dict[words[data_index]])
#                 data_index += 1
#             
#             context = list(compress(buffer, mask)) + [topic_dict[topic]]
#             target = buffer[-1]
#             all_instances.append(context)
#             all_labels.append(target)
#             
#             while data_index < len(words):
#                 buffer.append(word_dict[words[data_index]])
#                 data_index += 1
#                 
#                 context = list(compress(buffer, mask)) + [topic_dict[topic]]
#                 target = buffer[-1]
#                 all_instances.append(context)
#                 all_labels.append(target)
    
        
    batchs_num = len(all_instances) // topic_word_batch_size
    if len(all_instances) % topic_word_batch_size > 0:
        batchs_num += 1
        residual_num = topic_word_batch_size - len(all_instances) % topic_word_batch_size
        print("residual_num: ", residual_num)
        all_instances.extend(all_instances[:residual_num])   
        all_labels.extend(all_labels[:residual_num]) 
    print("batchs_num: ", batchs_num)
    print("all_instances samples: ", all_instances[0:10])
    print("all_labels samples: ", all_labels[0:10])
     
    all_instances = np.array(all_instances)
    all_labels = np.array(all_labels)
    shuffle_indices = np.random.permutation(np.arange(len(all_instances))) 
    all_instances = all_instances[shuffle_indices]
    all_labels = all_labels[shuffle_indices]
    training_batches = [(all_instances[i*topic_word_batch_size:(i+1)*topic_word_batch_size], 
                         all_labels[i*topic_word_batch_size:(i+1)*topic_word_batch_size]) for i in range(batchs_num)]
#     all_instances = np.array(all_instances)
#     all_labels = np.array(all_labels)
#     
#     batchs_num = len(all_instances) // topic_word_batch_size
#     if len(all_instances) % topic_word_batch_size > 0:
#         batchs_num += 1
#         residual_num = topic_word_batch_size - len(all_instances) % topic_word_batch_size
#         print("residual_num: ", residual_num)
#         indexes = random.sample(range(0, len(all_instances)), residual_num)   
#         all_instances = np.concatenate((all_instances, all_instances[indexes]), axis = 0)   
#         all_labels = np.concatenate((all_labels, all_labels[indexes]), axis = 0) 
#     print("batchs_num: ", batchs_num)
#     print("all_instances.shape: ", all_instances.shape)
#     print("all_labels.shape: ", all_labels.shape)
#     print("all_instances samples: ", all_instances[0:10])
#     print("all_labels samples: ", all_labels[0:10])
#     
#     training_batches = [(all_instances[i*topic_word_batch_size:(i+1)*topic_word_batch_size], 
#                          all_labels[i*topic_word_batch_size:(i+1)*topic_word_batch_size]) for i in range(batchs_num)]
    
    return training_batches, reverse_word_dict, reverse_topic_dict

# generate_batach_pvdm(URL.FILES.serviceData, 120, 3)    

def generate_batch_deepwalk(dir, node_node_batch_size, 
                            num_skip, node_window_size):
    print("Generating the training instances for DeepWalk model...")
    allnodes, _, _ = generateWalks(dir)
    node_dict, _ = build_dictionary(allnodes, 'network')
    print('node_dict:',len(node_dict))
    reverse_node_dict = dict(zip(node_dict.values(), node_dict.keys()))
    
    skip_window = int(node_window_size // 2)
    assert num_skip <= 2 * skip_window
    
    nn_all_instances = []
    nn_all_labels = []
    count = 0
    
    for nodetopic in allnodes:
        nodes = nodetopic.nodes
        
        for center_index in range(node_window_size, len(nodes)-(node_window_size+1)):
            buffer = []
            
            data_index = center_index
            for _ in range(skip_window):
                data_index = data_index - 1
                if data_index >= 0:
                    buffer.append(data_index)
                    
            data_index = center_index
            for _ in range(skip_window):
                data_index = data_index + 1
                if data_index <= (len(nodes) - 1):
                    buffer.append(data_index);
            
            object_to_avoid = []
            object_index = random.randint(0, len(buffer) - 1)
            for j in range(num_skip):
                if j >= len(buffer):
                    break
                while object_index in object_to_avoid:
                    object_index = random.randint(0, len(buffer) - 1)
                
                object_to_avoid.append(object_index)
               
                # node-node training instance
                nn_all_instances.append(node_dict[nodes[center_index]])
                nn_all_labels.append(node_dict[nodes[buffer[object_index]]])
                
        count += 1
        if count % 5000 == 0:
            gc.collect()
            
    print('nn_all_instances samples: ', nn_all_instances[0])
    print('nn_all_labels samples: ', nn_all_labels[0])
    
    nn_batchs_num = len(nn_all_instances) // node_node_batch_size
    if len(nn_all_instances) % node_node_batch_size > 0:
        nn_batchs_num += 1
        nn_residual_num = node_node_batch_size - len(nn_all_instances) % node_node_batch_size
        print("nn_residual_num: ", nn_residual_num)
        nn_all_instances.extend(nn_all_instances[:nn_residual_num])   
        nn_all_labels.extend(nn_all_labels[:nn_residual_num]) 
    print("nn_batchs_num: ", nn_batchs_num)
    
    nn_all_instances = np.array(nn_all_instances)
    nn_all_labels = np.array(nn_all_labels)
    shuffle_indices = np.random.permutation(np.arange(len(nn_all_instances))) 
    nn_all_instances = nn_all_instances[shuffle_indices]
    nn_all_labels = nn_all_labels[shuffle_indices]
    
    nn_training_batches = [(nn_all_instances[i*node_node_batch_size:(i+1)*node_node_batch_size], 
                         nn_all_labels[i*node_node_batch_size:(i+1)*node_node_batch_size]) for i in range(nn_batchs_num)]
    return nn_training_batches, reverse_node_dict

# def main():
#     nn_training_batches, reverse_node_dict = generate_batch_deepwalk(URL.FILES.coraData, 120, 2, 2)              
#     print(len(reverse_node_dict))
# #     print(reverse_node_dict[10309])
# if __name__ == '__main__':
#     main()    

def generate_batch_skipgram(dir, T, node_node_batch_size, topic_node_batch_size,
                            node_window_size, num_skip, topicN=1): 
    '''
    Batch generator for SkipGram model.
    batch should be a shape of (node_node_batch_size) or (topic_node_batch_size)
    
    Parameters
    ------------
    dir: dataset root directory
    node_node_batch_size: number of nodes in each mini-batch
    topic_node_batch_size: number of topic in each mini-batch
    word_window_size: number of context words to infer the target word
    '''
    print("Generating the training instances for SkipGram model...")
    
    allnodes, topic_size, node2topics = generateWalks(dir, T,number_walks=20, walk_length=10, topicN=topicN)
    
    node_dict, topic_dict = build_dictionary(allnodes, 'network')
    print('node_dict:',len(node_dict))
    reverse_node_dict = dict(zip(node_dict.values(), node_dict.keys()))
    reverse_topic_dict = dict(zip(topic_dict.values(), topic_dict.keys()))
    
    skip_window = int(node_window_size // 2)
    assert num_skip <= 2 * skip_window
    
    nn_all_instances = []
    nn_all_labels = []
    tn_all_instances = []
    tn_all_labels = []
    count = 0
    
    nodetopics = dict()
    for k in node2topics.keys():
        topics = []
        if not k in node_dict.keys():
            continue
        for t in node2topics[k]:
            topics.append(topic_dict[t])
        nodetopics[node_dict[k]] = topics
    
    for nodetopic in allnodes:
        nodes = nodetopic.nodes
        topics = nodetopic.topics

        
        for center_index in range(node_window_size, len(nodes)-(node_window_size+1)):
            buffer = []
            
            data_index = center_index
            for _ in range(skip_window):
                data_index = data_index - 1
                if data_index >= 0:
                    buffer.append(data_index)
                    
            data_index = center_index
            for _ in range(skip_window):
                data_index = data_index + 1
                if data_index <= (len(nodes) - 1):
                    buffer.append(data_index);
            
            object_to_avoid = []
            object_index = random.randint(0, len(buffer) - 1)
            for j in range(num_skip):
                if j >= len(buffer):
                    break
                while object_index in object_to_avoid:
                    object_index = random.randint(0, len(buffer) - 1)
                
                object_to_avoid.append(object_index)
               
                # node-node training instance
                nn_all_instances.append(node_dict[nodes[center_index]])
                nn_all_labels.append(node_dict[nodes[buffer[object_index]]])
                
                # topic-node training instance
                center_topics = np.array([topic_dict[top] for top in topics[center_index]])
                object_topics = np.array([topic_dict[top] for top in topics[buffer[object_index]]])
                center_onehot_topicvec = np.zeros([topic_size,1])
                object_onehot_topicvec = np.zeros([topic_size,1])
                center_onehot_topicvec[center_topics] = 1
                object_onehot_topicvec[object_topics] = 1
                shared_onehot_topicvec = np.multiply(center_onehot_topicvec,
                                                     object_onehot_topicvec)
                if np.sum(shared_onehot_topicvec) != 0:
                    tn_all_instances.append(shared_onehot_topicvec)
                    tn_all_labels.append(node_dict[nodes[buffer[object_index]]])
                
                center_onehot_topicvec = None
                object_onehot_topicvec = None
#         print(count)
        count += 1
        if count % 5000 == 0:
            gc.collect()
            
                    
    gc.collect()            
#     nn_all_instances = np.array(nn_all_instances)
#     nn_all_labels = np.array(nn_all_labels)
#     tn_all_instances = np.array(tn_all_instances)
#     tn_all_labels = np.array(tn_all_labels)
    print('nn_all_instances samples: ', nn_all_instances[0])
    print('nn_all_labels samples: ', nn_all_labels[0])
    print('tn_all_instances samples: ', tn_all_instances[0])
    print('tn_all_labels samples: ', tn_all_labels[0])
    
    nn_batchs_num = len(nn_all_instances) // node_node_batch_size
    if len(nn_all_instances) % node_node_batch_size > 0:
        nn_batchs_num += 1
        nn_residual_num = node_node_batch_size - len(nn_all_instances) % node_node_batch_size
        print("nn_residual_num: ", nn_residual_num)
        nn_all_instances.extend(nn_all_instances[:nn_residual_num])   
        nn_all_labels.extend(nn_all_labels[:nn_residual_num]) 
    print("nn_batchs_num: ", nn_batchs_num)
    
    
    nn_all_instances = np.array(nn_all_instances)
    nn_all_labels = np.array(nn_all_labels)
    shuffle_indices = np.random.permutation(np.arange(len(nn_all_instances))) 
    nn_all_instances = nn_all_instances[shuffle_indices]
    nn_all_labels = nn_all_labels[shuffle_indices]
    nn_training_batches = [(nn_all_instances[i*node_node_batch_size:(i+1)*node_node_batch_size], 
                         nn_all_labels[i*node_node_batch_size:(i+1)*node_node_batch_size]) for i in range(nn_batchs_num)]
    
    
    tn_batchs_num = len(tn_all_instances) // topic_node_batch_size
    if len(tn_all_instances) % topic_node_batch_size > 0:
        tn_batchs_num += 1
        tn_residual_num = topic_node_batch_size - len(tn_all_instances) % topic_node_batch_size
        print("tn_residual_num: ", tn_residual_num)
        tn_all_instances.extend(tn_all_instances[:tn_residual_num])   
        tn_all_labels.extend(tn_all_labels[:tn_residual_num]) 
    print("tn_batchs_num: ", tn_batchs_num)
    
    tn_all_instances = np.array(tn_all_instances)
    tn_all_labels = np.array(tn_all_labels)
    shuffle_indices = np.random.permutation(np.arange(len(tn_all_instances))) 
    tn_all_instances = tn_all_instances[shuffle_indices]
    tn_all_labels = tn_all_labels[shuffle_indices]
    tn_training_batches = [(tn_all_instances[i*topic_node_batch_size:(i+1)*topic_node_batch_size], 
                         tn_all_labels[i*topic_node_batch_size:(i+1)*topic_node_batch_size]) for i in range(tn_batchs_num)]
    
    return nn_training_batches, tn_training_batches, reverse_node_dict, nodetopics
    
#                 
# def main():
#     nn_training_batches, tn_training_batches, reverse_node_dict, nodetopics = generate_batch_skipgram(URL.FILES.coraData, 2, 2, 2, 2)               
#     print(len(reverse_node_dict))
# #     print(reverse_node_dict[10309])
# if __name__ == '__main__':
#     main()        

def repository_size(dir, T, topicN = 1):
    content_token = os.path.join(dir, "content_token.txt")
    labels = os.path.join(dir, 'topics'+str(topicN)+'.txt')
#     labels = os.path.join(dir, 'topics'+str(topicN)+'_T_'+ str(T)+'.txt')
    adjlist = os.path.join(dir, "adjlist.txt")
    
    word_idct = dict()
    label_idct = dict()
    node_idct = dict()
    with open(content_token, 'r', encoding="iso-8859-1") as cr, open(labels) as lr, open(adjlist) as ar:
        
        for l1 in cr:
            for w in l1.split():
                word_idct[w] = 1
        for l2 in lr: 
            for l in l2.split()[1].split(','):
                label_idct[l] = 1
        for l3 in ar:
            for n in l3.split():
                node_idct[n] = 1
    return len(node_idct), len(label_idct), len(word_idct)

# node_size, topic_size, word_size = repository_size(URL.FILES.coraData)  
# print('node_size:{}, topic_size:{}, word_size:{}'.format(node_size,topic_size,word_size))  
        
def convertDataFormat_cora(dir):
    
    datafile = os.path.join(dir, "data.txt")
    graphfile = os.path.join(dir, "graph.txt")
    groupfile = os.path.join(dir, "group.txt")
    
    content_token = os.path.join(dir, "content_token.txt")
    labels = os.path.join(dir, "labels.txt")
    adjlist = os.path.join(dir, "adjlist.txt")
    
    with open(datafile, 'r') as dr, open(content_token, 'w') as cw:
        for line in dr:
            cw.write(line.lower())
    
    adjnodes = dict()
    with open(graphfile, 'r') as gr:
        for line in gr:
            ns = line.split()
            
            if not ns[0] in adjnodes.keys():
                adjnodes[ns[0]] = [ns[1]]
            elif not ns[1] in adjnodes[ns[0]]:
                adjnodes[ns[0]].append(ns[1])
            
            if not ns[1] in adjnodes.keys():
                adjnodes[ns[1]] = [ns[0]]
            elif not ns[0] in adjnodes[ns[1]]:
                adjnodes[ns[1]].append(ns[0])
    with open(adjlist, 'w') as aw:
        
        for k in adjnodes.keys():
            aw.write(k + ' ' + ' '.join(adjnodes[k]) + '\n')
            
    with open(groupfile, 'r') as gr, open(labels, 'w') as lw:
        
        label_dict = dict()
        line_count = 0
        for line in gr:
            labs = line.split()
            if len(labs) != 0:
                if not labs[0] in label_dict:
                    label_dict[labs[0]] = 'L' + str(len(label_dict))
                lw.write(str(line_count) + ' ' + label_dict[labs[0]] + '\n')
            
            line_count += 1
                
    
# convertDataFormat_cora(URL.FILES.coraData)    

def convertDataFormat_citeseer(dir):
    
    docfile = os.path.join(dir, "citeseer.features")
    adjedgesfile = os.path.join(dir, "citeseer_edgelist.txt")
    labelsfile = os.path.join(dir, "groups.txt")
    
    content_token = os.path.join(dir, "content_token.txt")
    labels = os.path.join(dir, "labels.txt")
    adjlist = os.path.join(dir, "adjlist.txt")
    
    docid_dict = dict()
    with open(docfile, 'r') as dr, open(content_token, 'w') as cw:
        for line in dr:
            params = line.split()
            
            words = ['w'+str(index) for index, value in enumerate(params[1:]) if float(value) != 0]
#             if len(words) == 0:
#                 continue
            docid_dict[params[0]] = len(docid_dict)
            
            cw.write(' '.join(words)+"\n")
            
    label_dict = dict()
    with open(labelsfile, 'r') as lr, open(labels, 'w') as lw:
         
        for line in lr:
            params = line.split()
#             if not params[0] in docid_dict.keys():
#                 continue
            nodeid = docid_dict[params[0]]
            if not params[1] in label_dict.keys():
                label_dict[params[1]] = 'L' + str(len(label_dict))
            lw.write(str(nodeid) + ' ' + label_dict[params[1]] + '\n')
            
    adjnodes = dict()
    with open(adjedgesfile, 'r') as gr:
        for line in gr:
            ns = line.split()
            
            if not ns[0] in docid_dict.keys() or not ns[1] in docid_dict.keys():
                continue
            
            ns[0] = str(docid_dict[ns[0]])
            ns[1] = str(docid_dict[ns[1]])
            
            if not ns[0] in adjnodes.keys():
                adjnodes[ns[0]] = [ns[1]]
            elif not ns[1] in adjnodes[ns[0]]:
                adjnodes[ns[0]].append(ns[1])
            
            if not ns[1] in adjnodes.keys():
                adjnodes[ns[1]] = [ns[0]]
            elif not ns[0] in adjnodes[ns[1]]:
                adjnodes[ns[1]].append(ns[0])
    with open(adjlist, 'w') as aw:
        
        for k in adjnodes.keys():
            aw.write(k + ' ' + ' '.join(adjnodes[k]) + '\n')
            
# convertDataFormat_citeseer(URL.FILES.citeseerData)     
            
def convertDataFormat_wiki(dir):
    
    docfile = os.path.join(dir, "citeseer.features")
    adjedgesfile = os.path.join(dir, "citeseer_edgelist.txt")
    labelsfile = os.path.join(dir, "groups.txt")
    
    content_token = os.path.join(dir, "rtm_content_token.txt")
    labels = os.path.join(dir, "rtm_labels.txt")
    adjlist = os.path.join(dir, "rtm_adjlist.txt")
    
    docid_dict = dict()
    with open(docfile, 'r') as dr, open(content_token, 'w') as cw:
        for line in dr:
            params = line.split()
            
            words = ['w'+str(index) for index, value in enumerate(params[1:]) if float(value) != 0]
            if len(words) == 0:
                continue
            docid_dict[params[0]] = len(docid_dict)
            
            cw.write(' '.join(words)+"\n")
            
    label_dict = dict()
    with open(labelsfile, 'r') as lr, open(labels, 'w') as lw:
         
        for line in lr:
            params = line.split()
            if not params[0] in docid_dict.keys():
                continue
            nodeid = docid_dict[params[0]]
            if not params[1] in label_dict.keys():
                label_dict[params[1]] = 'L' + str(len(label_dict))
            lw.write(str(nodeid) + ' ' + label_dict[params[1]] + '\n')
            
    adjnodes = dict()
    with open(adjedgesfile, 'r') as gr:
        for line in gr:
            ns = line.split()
            
            if not ns[0] in docid_dict.keys() or not ns[1] in docid_dict.keys():
                continue
            
            ns[0] = str(docid_dict[ns[0]])
            ns[1] = str(docid_dict[ns[1]])
            
            if not ns[0] in adjnodes.keys():
                adjnodes[ns[0]] = [ns[1]]
            elif not ns[1] in adjnodes[ns[0]]:
                adjnodes[ns[0]].append(ns[1])
            
            if not ns[1] in adjnodes.keys():
                adjnodes[ns[1]] = [ns[0]]
            elif not ns[0] in adjnodes[ns[1]]:
                adjnodes[ns[1]].append(ns[0])
    with open(adjlist, 'w') as aw:
        
        for k in adjnodes.keys():
            aw.write(k + ' ' + ' '.join(adjnodes[k]) + '\n')
   
# convertDataFormat_wiki(URL.FILES.citeseerData)    
    
def extracttopics(dir, topicN, T):
    
    thetafile= os.path.join(dir,'T_'+ str(T) +'_theta.txt')
    topicfile= os.path.join(dir,'topics'+str(topicN)+'_T_'+ str(T)+'.txt')
    with open(thetafile) as tr, open(topicfile, 'w') as tw:
        unique_topics = dict()
        linecount = 0
        for line in tr:
            topics = [float(value) for value in line.split()]
            topics = sorted(range(len(topics)), key=lambda i:topics[i], reverse=True)[:topicN]
            
            atopics = []
            for topic in topics:
                if not topic in unique_topics:
                    unique_topics[topic] = len(unique_topics)
                atopics.append("L" + str(unique_topics[topic]))
            wline = str(linecount) + ' ' + ','.join(atopics)
            tw.write(wline+'\n')
            
            linecount += 1

# extracttopics(URL.FILES.coraData, 2, 35) 
