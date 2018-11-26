# -*- coding: utf-8 -*-

# Network Data pre-prcesssing

import URLs as URL
import os
import csv

def buildServiceNet(apiFile, mashupFile):
    # output files
    cntFile = os.path.join(URL.FILES.serviceData,"content_new.txt")
    graphFile = os.path.join(URL.FILES.serviceData, "adjlist.txt")
    labelFile = os.path.join(URL.FILES.serviceData, "labels.txt")
    labelnameFile = os.path.join(URL.FILES.serviceData, "labelNames.txt")
    
    contents = []
    labels = dict()
    links = dict()
    
    labelist = []
    apinamelist = []
    
    validapis = []
    with open(mashupFile, newline='', encoding='iso-8859-1') as mashupcsv:
        spamreader = csv.reader(mashupcsv, delimiter=',')
        header = next(spamreader, None)
        for row in spamreader:
            mashup_apis = str.lower(row[3]).split(",")
            for apin in mashup_apis:
                apin = str.strip(apin)
                if not apin in validapis:
                    validapis.append(apin)
    print("validapis:",len(validapis))
    
    with open(apiFile, newline='', encoding='iso-8859-1') as apicsv:
        spamreader = csv.reader(apicsv, delimiter=',')
        header = next(spamreader, None)
        for row in spamreader:
            api_name = str.lower(row[0])
            if not api_name in validapis:
                print(api_name)
                continue
            
            api_tags = str.lower(row[1]).split("###")
            api_content = str.lower(row[2])
#             api_content.replace('\n',' ')
            api_content = api_content.replace('\n', ' ').replace('\r', '')
            
            apinamelist.append(api_name)
            contents.append(api_content)
            
            labs = []
            if not api_tags[0] in labelist:
                labelist.append(api_tags[0])
#             for tag in api_tags:
#                 if not tag in labelist:
#                     labelist.append(tag)
#                 if not "L"+str(labelist.index(tag)) in labs:
#                     labs.append("L"+str(labelist.index(tag)))
            labs.append("L"+str(labelist.index(api_tags[0])))
            labels[len(contents)-1] = labs
            
    print("Total number of labels:", len(labelist))
    print("contents:", len(contents))
    print("links:", len(links))
    print("labels:", len(labels))
    print("\n")
    linkcount = 0
    with open(mashupFile, newline='', encoding='iso-8859-1') as mashupcsv:
        spamreader = csv.reader(mashupcsv, delimiter=',')
        header = next(spamreader, None)
        for row in spamreader:
            mashup_name = str.lower(row[0])
            mashup_tags = str.lower(row[1]).split("###")
            mashup_content = str.lower(row[2])
            mashup_apis = str.lower(row[3]).split(",")
            
            contents.append(mashup_content)
            
            labs = []
#             for tag in mashup_tags:
#                 if not tag in labelist:
#                     labelist.append(tag)
            if not mashup_tags[0] in labelist:
                labelist.append(mashup_tags[0])
#                 if not "L"+str(labelist.index(tag)) in labs:
#                     labs.append("L"+str(labelist.index(tag)))
            labs.append("L"+str(labelist.index(mashup_tags[0])))
            labels[len(contents)-1] = labs
            
            apils = []
            mashupindex = len(contents)-1
            for apin in mashup_apis:
                apin = str.strip(apin)
                if apin in apinamelist:
                    apiindex = apinamelist.index(apin)
                    if not apiindex in links.keys():
                        links[apiindex] = []
                    links[apiindex].append(str(mashupindex))
                    apils.append(str(apiindex))
                    linkcount += 1
            links[mashupindex] = apils
            
                
#     with open(labelnameFile, 'w') as lnw:
#         for lname in labelist:
#             lnw.write(lname+"\n")
            
    print("Total number of labels:", len(labelist))
    print("contents:", len(contents))
    print("links:", len(links))
    print("labels:", len(labels))
    print("linkcount:", linkcount)
    
    with open(cntFile, 'w',encoding='iso-8859-1') as cw:
        for i in  range(len(contents)):
            cw.write(contents[i]+"\n")
            
    with open(labelFile, 'w',encoding='iso-8859-1') as lw:
        for key in labels:
            lw.write(str(key) +" " + ','.join(labels[key]) + "\n")
            
    with open(graphFile, 'w',encoding='iso-8859-1') as gw:
        for key in links:
            gw.write(str(key) +" " + ' '.join(links[key]) + "\n")
            
    with open(labelnameFile, 'w',encoding='iso-8859-1') as lnw:
        for i in range(len(labelist)):
            lnw.write("L" + str(i) + ' ' + labelist[i] + "\n")   

# buildServiceNet(os.path.join(URL.FILES.serviceData,"raw_APIs.csv"), os.path.join(URL.FILES.serviceData,"raw_Mashups.csv"))

def data_PLANE(data, destination, dataset_flag):
    
    contentfile = os.path.join(data,"content_token.txt")
    adjlistfile = os.path.join(data, "adjlist.txt")
    
    docfile = os.path.join(destination,dataset_flag+"_doc.txt")
    linkfile = os.path.join(destination,dataset_flag+"_link.txt")
    
    doc_count = 0
    with open(contentfile, 'r', encoding='iso-8859-1') as cr, open(docfile, 'w') as aw:
        for line in cr:
            aw.write(line)
            doc_count += 1
    
    links = dict()
    with open (adjlistfile) as lr:
        for line in lr:
            params = line.split()
            links[int(params[0])] = [int(value) for value in params[1:]]
            
    matrix = dict()
    for i in range(doc_count):
        ids = ['0'] * doc_count
        if i in links.keys():
            for j in links[i]:
                ids[j] = '1'
        matrix[i] = ids
    print(matrix.keys())
    with open(linkfile, 'w') as lw:
        for k in matrix.keys():
            line = ' '.join(matrix[k])
            lw.write(line+'\n')

# data_PLANE(URL.FILES.coraData, 'F:\\sharedhome\\PLANE-master', 'cora')
                
def label_statistic(data):
    cntFile = os.path.join(data,"content_token.txt")
    graphFile = os.path.join(data, "adjlist.txt")
    labelFile = os.path.join(data, "labels.txt")
    
    cntFile_freq = os.path.join(data,"content_token_freq.txt")
    graphFile_freq = os.path.join(data, "adjlist_freq.txt")
    labelFile_freq = os.path.join(data, "labels_freq.txt")
    
    labels = dict()
    selected_labels = []
    with open(labelFile) as lr:
        for line in lr:
            params = line.split()
            if not params[1] in labels.keys():
                labels[params[1]] = [params[0]]
            else:
                labels[params[1]].append(params[0])
    label_dict = dict()
    for key, value in sorted(labels.items(), key=lambda x: len(x[1]), reverse=True):
        if len(value) >= 50:
            if not key in label_dict.keys():
                label_dict[key] = 'L'+str(len(label_dict))
            selected_labels.append(key)
    print('selected_labels:',len(selected_labels))
    
    node_ids = dict()
    node_ids_reverse = dict()
    with open(labelFile) as lr, open(labelFile_freq, 'w') as lw:
        for line in lr:
            params = line.split()
            if params[1] in selected_labels:
                node_ids[params[0]] = len(node_ids)
                node_ids_reverse[len(node_ids)] = int(params[0])
                lw.write(str(node_ids[params[0]]) + ' ' + label_dict[params[1]] + '\n')
    print(node_ids_reverse)
    conts = dict()
    with open(cntFile, 'r', encoding='iso-8859-1') as cr, open(cntFile_freq, 'w', encoding='iso-8859-1') as cw:
        
        lcount = 0
        for line in cr:
            conts[lcount] = line;
            lcount += 1
        
        for k in node_ids_reverse.keys():
            cw.write(conts[node_ids_reverse[k]])
            
    with open(graphFile) as gr, open(graphFile_freq, 'w') as gw:
        
        for line in gr:
            id = line.split()[0]
            linked_ids = line.split()[1:]
            
            if id in node_ids.keys():
                id_new = str(node_ids[id])
                
                linked_ids_new = ' '.join([str(node_ids[value]) for value in linked_ids if value in node_ids.keys()])
                
                gw.write(id_new + ' ' + linked_ids_new + '\n')
            
# label_statistic(URL.FILES.serviceData)            
            
            
    