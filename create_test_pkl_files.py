import collections,pickle
from collections import defaultdict
from typing import runtime_checkable
col_dic=defaultdict(list)

collection_file=open('collection/collection.tsv','r').readlines()

for line in collection_file:
    docid,doctext= line.rstrip().split('\t')
    col_dic[docid]=doctext
q_file= open ('queries.dev.small.tsv','r').readlines()

q_map_dic={}


for line in q_file:
    qid,qtext=line.rstrip().split('\t')
    q_map_dic[qid]={}
    q_map_dic[qid] ["qtext"]=qtext
    
run_file=open('run/bm25_first_docs_dev.tsv','r').readlines()
for line in run_file:
    qid,docid,rank=line.split('\t')
    if qid in q_map_dic.keys():
        q_map_dic[qid]["doc_text"]=col_dic[docid]


with open('pklfiles/test_dev_map.pkl', 'wb') as f:
    pickle.dump(q_map_dic, f, pickle.HIGHEST_PROTOCOL)
