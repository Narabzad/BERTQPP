from sentence_transformers import SentenceTransformer, InputExample, losses, util,evaluation
from torch.utils.data import DataLoader
import pickle 
from scipy.stats import kendalltau,pearsonr

with open('pklfiles/train_map.pkl', 'rb') as f:
    q_map_first_doc_train=pickle.load(f)

train_examples=[]

for key in q_map_first_doc_train:
    qtext=q_map_first_doc_train[key]["qtext"]
    firstdoctext=q_map_first_doc_train[key]["doc_text"]
    map_value=q_map_first_doc_train[key]["performance"]
    train_examples.append( InputExample(texts=[qtext,firstdoctext],label=map_value ))


batch_size=8
num_epoch=1
model = SentenceTransformer('bert-base-uncased')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epoch, warmup_steps=100)
model.save("models/tuned_model_bi_bert-base-uncased_e_"+str(num_epoch)+'_b_'+str(batch_size))

