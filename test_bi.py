from sentence_transformers import SentenceTransformer, InputExample, losses, util,evaluation
from torch.utils.data import DataLoader
import pickle 
from scipy.stats import kendalltau,pearsonr

with open('pklfiles/test_dev_map.pkl', 'rb') as f:
    q_map_first_doc_test=pickle.load(f)

model_name="tuned_model_bi_bert-base-uncased_e_1_b_8"

sentences1 = []
sentences2 = []
map_value_test=[]
qs=[]

for key in q_map_first_doc_test:
    sentences1.append(q_map_first_doc_test[key]["qtext"])
    sentences2.append(q_map_first_doc_test[key]["doc_text"])
    qs.append(key)

model=SentenceTransformer("models/"+model_name)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

actual=[]
predicted=[]
out=open('results/QPP-bi_'+model_name,'w')

for i in range(len(sentences1)):
    out.write(qs[i]+'\t'+str(float(cosine_scores[i][i]))+'\n')
out.close()
