from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import os
import gensim
import URLs as URL
import numpy as np



def build_dataset(dir):
    docs = []
    
    contentfile = os.path.join(dir,"content_token.txt")
    
    with open(contentfile, 'r') as cr:
        
        for line in cr:
#             line = gensim.parsing.stem_text(line)
            docs.append(line)
    print(docs[0:2])
    return docs

def LDAtrain(dir, k, max_iter):
    
    docs = build_dataset(dir)
    
    tf_vectorizer  = CountVectorizer(max_df=0.85, min_df=2, stop_words='english')
    tf = tf_vectorizer .fit_transform(docs)
    
    vocab = tf_vectorizer.get_feature_names()
    
    lda = LatentDirichletAllocation(n_topics=k, max_iter=max_iter, learning_method='online', 
                                    learning_offset=50.,random_state=100).fit(tf)
                                    
    theta = lda.transform(tf)
    
    thetafile= os.path.join(dir,'sklearn_theta.txt')
    with open(thetafile, 'w') as tw:
        for doc_dist in theta:
            str_dist = [str(value) for value in doc_dist]
            tw.write(' '.join(str_dist) + '\n')
    
    return lda, vocab

def extractTopics(dir, topicN):
    
    thetafile= os.path.join(dir,'sklearn_theta.txt')
    topicfile= os.path.join(dir,'sk_topics'+str(topicN)+'.txt')
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
            
def sampletopics(model, vocab):
    topic_words = {}
    n_top_words = 20
    for topic, comp in enumerate(model.components_):
        word_idx = np.argsort(comp)[::-1][:n_top_words]
        # store the words most relevant to the topic
        topic_words[topic] = [vocab[i] for i in word_idx]
     
    for topic, words in topic_words.items():
        print('Topic: %d' % topic)
        print('  %s' % ', '.join(words))

def main():
    dataset = URL.FILES.coraData
    k = 20
    max_iter = 100
    lda, vocab = LDAtrain(dataset, k, max_iter)
    sampletopics(lda, vocab)
    
    topicN = 4
    extractTopics(dataset, topicN)

if __name__ == '__main__':
    main()
