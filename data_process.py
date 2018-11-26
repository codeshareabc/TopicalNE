# -*- coding: utf-8 -*-
from _overlapped import NULL

import collections
import os
import gensim
import random
import gensim.utils as ut
import URLs as URL
import numpy as np
from collections import namedtuple


def process_coradataset(source, target):
    
    featurefile = os.path.join(source, "content_token.txt")
    edgelistfile = os.path.join(source, "graph.txt")
    labelsfile = os.path.join(source, "group.txt")
    
    content = os.path.join(target, "content.txt")
    labels = os.path.join(target, "labels.txt")
    adjlist = os.path.join(target, "adjlist.txt")

    word_dict = dict()
    word_ids = []
    with open(featurefile, 'r') as fr, open(content, 'w') as wc:
        for line in fr:
            ids = []
            line = gensim.parsing.stem_text(line)
            words = line.split()
            for word in words:
                if not word in word_dict.keys():
                    word_dict[word] = len(word_dict)
                if not word_dict[word] in ids:
                    ids.append(word_dict[word])
            word_ids.append(ids)
            
        linecount = 0
        for doc in word_ids:
            
            vocabulary = np.zeros(len(word_dict))
            vocabulary[doc] = 1.0
            
            word_binary = [str(value) for value in vocabulary]
            wline = str(linecount) + ' ' + ' '.join(word_binary)
            wc.write(wline+'\n')
            linecount += 1
    
    edjes = dict()
    with open(edgelistfile, 'r') as er, open(adjlist, 'w') as aw:
        
        for line in er:
            if not int(line.split()[0]) in edjes.keys():
                edjes[int(line.split()[0])] = [line.split()[1]]
            else:
                edjes[int(line.split()[0])].append(line.split()[1])
            
        for key in sorted(edjes.keys()):
            for la in edjes[key]:
                wline = str(key) + ' ' + la
                aw.write(wline+'\n')
                
    with open(labelsfile, 'r') as lr, open(labels, 'w') as lw:
        
        linecount = 0
        for line in lr:
            if line.strip() == '':
                linecount += 1
                continue
            wline = str(linecount) + ' ' + line
            lw.write(wline)
            linecount += 1
        
# process_coradataset(URL.FILES.coraData, 'F:\\git\\OpenNE\\data\\cora')   
        
def dataset_statistics(dir):
    
    featurefile = os.path.join(dir, "content_token.txt")    
    voca = []  
    words = []
    linecount = 0
    with open(featurefile, 'r') as fr:
        for line in fr:
            line = line.split()
            
            for word in line:
                if not word in voca:
                    voca.append(word)
            
            words.extend(line)
      
            linecount += 1
    print('Vocabulary size:', len(voca))
    print('Average word per doc:', len(words)/linecount)
        
dataset_statistics(URL.FILES.citeseerData)        
        