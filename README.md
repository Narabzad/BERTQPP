# BERT-QPP: Contextualized Pre-trained Transformers for Query Performance Prediction

In this paper, we adopt contextual embeddings to perform performance prediction specifically for the task of query performance prediction.The fine-tuned contextual representations can estimate the performance of a query based on the association between the representation of the query and the retrieved documents. We compare the performance of our approach with the state-of-the-art based on the MS MARCO passage retrieval corpus and its three associated query sets: (1) MS MARCO development set, (2) TREC DL 2019, and (3) TREC DL 2020. We show that our approach not only shows significant improved prediction performance compared to all the state-of-the-art methods, but also, unlike past neural predictors, it shows significantly lower latency, making it possible to use in practice.

We adopt two architechtures namely cross-encoder network and bi-encoder network to address QPP task. 

To replicate our results  with BERT-QPP<sub>cross</sub> and BERT-QPP<sub>bi</sub> on MSMARCO passage collection,

 1. Clone this repository.
 2. Install the required packages are listed in ```requirement.txt``` on python 3.7+. 
 3. Download [MSMARCO collection](https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz) ```collection.tsv``` and store it in ```collection``` repository.
 4. If you are willing to predict the performance of BM25 retrieval method on MSMARCO, skip this step. Otherwise, when evaluating any other retrieval method, you need to prepare the similar run file to ```bm25_first_docs_train.tsv``` and ```bm25_first_docs_dev.tsv``` which include the run file for first retrieved documents for queries in MSMARCO train and dev set. 
    * The runfile of your desired retrieval approach should havethe folloinwg format for each query per line:  ```QID<\t>DOCID<\t>1```. 
    * Then, modify the ```run_file``` variable in ```create_train_pkl_file.py``` and ```create_test_pkl_file.py``` so that they point to your desired ```run_file```s on train and sev set of MSMARCO.
 5. To train BERT-QPP<sub>cross</sub>, we require the query, the first retrieved document, and the queries' performance. To do so,  in ```create_train_pkl_file.py``` we create a dictionary including the following attributes:
```
    train_dic[qid] ["qtext"]=query_text
    train_dic[qid] ["performance"]=query_performance_value
    train_dic[qid]["doc_text"]=document_text
 ```
   you can train the model on your desired metric by creating the assosiated train pkl file. Here, we use map@20.
   Run ```create_train_pkl_file.py``` to save a dictionary including query and document text as well as their associated performance. As a result ```train_map.pkl``` will be saved in ```pklfiles``` directory.

 7. Run ```create_test_pkl_file.py``` to save a dictionary including query and document text on the MSMARCO developement set. As a result ```test_dev_map.pkl``` will be saved in ```pklfiles``` directory.


## BERT-QPP<sub>cross</sub>
 1. run ```train_CE.py``` to learn the map@20 of BM25 retrieval on MSMARCO train set. alternatively, you can train with your desired metric by creating the assosiated train pkl file. me On a single 24GB RTX3090 GPU, it took less than 2 hours. You may also change the ```epoch_num```,```batch_size```, and initial  pre-trained model in this file. We used ```bert-base-uncased``` in this experiment. The trained model will be saved in ```models``` directory.
 2. If you are not willing to train the model, you can download our BERT-QPP<sub>cross</sub> [trained model on bert-based-uncased from here](https://drive.google.com/drive/folders/1NDZzEpaay0cDumTKDUSMmv99sg9FyHrL?usp=sharing).
 3. add the ```trained_model``` you are willing to test in ```test_CE.py``` and  run ```test_CE.py```.
 4. The results will be saved in results directory in the following format: QID\tPredicted_QPP_value
The results will be saved in ```results``` directory in the following format:
    ```QID<\t>Predicted_QPP_value```
 5. To evaluate the results, you can calculate the correlation between the actual performance of each query and predicted QPP value.

## BERT-QPP<sub>bi</sub>
 1. run ```train_bi.py``` to learn the map@20 of BM25 retrieval on MSMARCO train set. . me On a single 24GB RTX3090 GPU, it took ~1hour. You may also change the ```epoch_num```,```batch_size```, and initial  pre-trained model in this file. We used ```bert-base-uncased``` in this experiment. The trained model will be saved in ```models``` directory.
 2. If you are not willing to train the model, you can download our BERT-QPP<sub>bi</sub> [trained model on bert-based-uncased from here](https://drive.google.com/drive/folders/1Kd3GK9yiJ3gulre8k-gvhNbmUSR7e324?usp=sharing).
 3. add the ```trained_model``` you are willing to test in ```test_bi.py``` and  run ```test_bi.py```.
 4. The results will be saved in results directory in the following format: QID\tPredicted_QPP_value
The results will be saved in ```results``` directory in the following format:
    ```QID\tPredicted_QPP_value```
 5. To evaluate the results, you can calculate the correlation between the actual performance of each query and predicted QPP value.

