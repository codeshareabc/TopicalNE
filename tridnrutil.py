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

def loadNetworkContent(dir, stemmer=0):
    alldocs = []
    contentfile = os.path.join(dir,"content_token.txt")
    labelfile = os.path.join(dir,"labels.txt")
    
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

def generateWalks(dir, number_walks=50, walk_length=10):
    adjlistfile = os.path.join(dir,"adjlist.txt")
    labelfile = os.path.join(dir,"labels.txt")
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

def generate_batch_triDNR(dir, node_word_batch_size, group_word_batch_size, train):
    print("Generating the training instances for TriDNR model...")
    
    alldocs = loadNetworkContent(dir, 0)
    print('alldocs:',len(alldocs))
    word_dict, topic_dict = build_dictionary(alldocs, 'content')
    reverse_word_dict = dict(zip(word_dict.values(), word_dict.keys()))
    reverse_group_dict = dict(zip(topic_dict.values(), topic_dict.keys()))
    
    allnodes, _, _ = generateWalks(dir)
    node_dict, _ = build_dictionary(allnodes, 'network')
    
    nw_all_instances = []
    nw_all_labels = []
    gw_all_instances = []
    gw_all_labels = []
    
    
    
    for wordtopic in alldocs:
        groups = wordtopic.topics
        words = wordtopic.words
        docId = wordtopic.docId
        
        if not docId in node_dict.keys():
            continue
        
        data_index = 0
        while data_index < len(words):
            nw_all_instances.append(node_dict[docId])
            nw_all_labels.append(word_dict[words[data_index]])
            
            if node_dict[docId] in train:
                for group in groups:
                    gw_all_instances.append(topic_dict[group])
                    gw_all_labels.append(word_dict[words[data_index]])
            data_index += 1
    
    
    nw_batchs_num = len(nw_all_instances) // node_word_batch_size
    if len(nw_all_instances) % node_word_batch_size > 0:
        nw_batchs_num += 1
        nw_residual_num = node_word_batch_size - len(nw_all_instances) % node_word_batch_size
        print("nw_residual_num: ", nw_residual_num)
        nw_all_instances.extend(nw_all_instances[:nw_residual_num])   
        nw_all_labels.extend(nw_all_labels[:nw_residual_num]) 
    print("nw_batchs_num: ", nw_batchs_num)
     
    nw_all_instances = np.array(nw_all_instances)
    nw_all_labels = np.array(nw_all_labels)
    shuffle_indices = np.random.permutation(np.arange(len(nw_all_instances))) 
    nw_all_instances = nw_all_instances[shuffle_indices]
    nw_all_labels = nw_all_labels[shuffle_indices]
    nw_training_batches = [(nw_all_instances[i*node_word_batch_size:(i+1)*node_word_batch_size], 
                         nw_all_labels[i*node_word_batch_size:(i+1)*node_word_batch_size]) for i in range(nw_batchs_num)]
    
    gw_batchs_num = len(gw_all_instances) // group_word_batch_size
    if len(gw_all_instances) % group_word_batch_size > 0:
        gw_batchs_num += 1
        tn_residual_num = group_word_batch_size - len(gw_all_instances) % group_word_batch_size
        print("tn_residual_num: ", tn_residual_num)
        gw_all_instances.extend(gw_all_instances[:tn_residual_num])   
        gw_all_labels.extend(gw_all_labels[:tn_residual_num]) 
    print("tn_batchs_num: ", gw_batchs_num)
    
    gw_all_instances = np.array(gw_all_instances)
    gw_all_labels = np.array(gw_all_labels)
    shuffle_indices = np.random.permutation(np.arange(len(gw_all_instances))) 
    gw_all_instances = gw_all_instances[shuffle_indices]
    gw_all_labels = gw_all_labels[shuffle_indices]
    gw_training_batches = [(gw_all_instances[i*group_word_batch_size:(i+1)*group_word_batch_size], 
                         gw_all_labels[i*group_word_batch_size:(i+1)*group_word_batch_size]) for i in range(gw_batchs_num)]

    return nw_training_batches, gw_training_batches, reverse_group_dict, reverse_word_dict

# def main():
#     generate_batch_triDNR(URL.FILES.coraData, 120, 120)
# if __name__ == '__main__':
#     main()



def repository_size(dir):
    content_token = os.path.join(dir, "content_token.txt")
    labels = os.path.join(dir, "labels.txt")
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
    return len(node_idct), len(label_idct), len(word_idct), 

# node_size, topic_size, word_size = repository_size(URL.FILES.coraData)  
# print('node_size:{}, topic_size:{}, word_size:{}'.format(node_size,topic_size,word_size))  