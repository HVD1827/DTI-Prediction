# DTIOG
Enhancing Drug-Target Interaction Prediction: A Graph-Based Approach Integrating Knowledge Graph Embedding and Pretrained ProtBert Model.

This tools presents a knowledge-graph (KG)-based, deep learning approach to discover potential drug-target interaction. Specifically,
the approach is based on drug repositioning from the association between different phenotypic, genomic, diseases, and cellular profiles. After applying a knowledge graph embedding on a heterogeneous graph to learn feature representation for each drug and target, cosine similarity is calculated between each drug and target to construct new matrices similarities. 

The main innovation of our approach is based on the flagship idea of building a knowledge graph on Homo-sapiens by combining many sources of information from public datasets related to drug discovery. Then, this information base will be the input of an algorithm based on the deep learning technique for predicting new interactions between drugs and targets.

Authors: Warith Eddine DJEDDI, Khalil HERMI, Sadok BEN YAHIA and GAYO DIALLO

Email: waritheddine.jeddi@isikef.u-jendouba.tn

# Quick start
To reproduce our results:
1. Untar the file "DTIOG-master.tar.xz"
2. Run <code>pred_new_dti.py</code> to reproduce the cross validation results of DTIOG and predict drug-target interaction using a given classifier (e.g., ExtraTreesClassifier, DecisionTreeClassifier, MLPClassifier or RandomForestClassifier).
# Data description
The whole dataset contains many part:
- "KGE/data/knowgraph_all.tsv", a tsv file containing the knowledge graph in the format of (h, r, t) triplets.

- folder containing the pretrained Knowledge Graph Embedding using the entire "KGE/data/knowgraph_all.tsv". It is worthy of mention, that the pretrained Knowledge Graph Embedding can be obtained by following these steps:

  a) Execute the python file "KGE/Train_data.py" in order to split the dile to "KGE/data/knowgraph_all.tsv" file in training, validation and testing data.
  
  b) Execute this command in order to generate the four files:
  
```
DGLBACKEND=pytorch dglke_train --dataset DTIOG --model_name DistMult --batch_size 2000 --neg_sample_size 1500 --hidden_dim 400 --gamma 19.9 --lr 0.25 --max_step 300000 --log_interval 100 --batch_size_eval 16 -adv --regularization_coef 1.00E-09 --test --data_path ./train/ --format raw_udd_hrt --data_files data_train.tsv data_valid.tsv data_test.tsv --neg_sample_size_eval 10000
```
there are four generated files:

     1) DTIOG_DistMult_entity.npy, NumPy binary data, storing the entity embedding
     
     2) DTIOG_DistMult_relation.npy, NumPy binary data, storing the relation embedding
     
     3) entities.tsv, mapping from entity_name to entity_id
     
     4) relations.tsv, mapping from relation_name to relation_id
     
   To use the pretrained embedding, one can use np.load to load the entity embeddings and relation embeddings separately:

```
import numpy as np
entity_emb = np.load('../data/ckpts/DTIOG_DistMult_entity.npy')
rel_emb = np.load('../data/ckpts/DTIOG_DistMult_relation.npy')
```
- For example, execute this command in order to generate the embedding similarities between the gpcr drugs using the cosinus metric:  
```
dglke_emb_sim --emb_file data/ckpts/DTIOG_DistMult_entity.npy --format 'l_r' --data_files data/train/source_drug_gpcr.list data/train/source_drug_gpcr.list --sim_func cosine --topK 188 --exec_mode 'batch_left'
```
Execute the 'entity_pairs_distibution.py' file in order to generate the two files "source_drug_gpcr.list" and "target_gpcr.list" required to perform the embedding similartities.
Execute the 'adjacence_matrix.py' file in order to convert the similarities file between drug and target to an adjacency matrix.
Execute the 'dtiog-protbertembedding.ipynb' file under the folder 'from_fasta_to_probert_embedding' in order to convert the FASTA format of the proteins to ProtBert embedding vectors.
Execute the 'embedding-similarities-drug-target-protbert-gpcr-ic-enz.ipynb' file under the folder 'KGE_PROTBERT_EMBEDDING' to compute the different similarity metrics (e.g., Manhattan distance, Cosine similarity, Jaccard Similarity, etc.) between drugs and proteins based on a given embedding strategy (KGE, ProtBERT, Molecular fingerprint or protein characteristics).
# Requirements
Python >= 3.8
# Inputs
* drug_list.txt: list of drug names.
* target_list.txt: list of target names.
* drug_drug_sim.txt : Drug-Drug similarities matrix.
* target_target_sim.txt : Target-Target similarities matrix.
* drug_target_association.txt : Drug-target association matrix.
* Drug_Proteins_Similarity_Matrix.txt : Drug-proteins similarities matrix.
* Drug_SideEffect_Similarity_Matrix.txt : Drug-Side Effects similarities matrix.
* enz_KGE_drug.dat: Vector embedding of enzyme drugs
* enz_KGE_target.dat: Vector embeddings of targets that interact with enzyme drugs.
* enz_KG.dat: Bipartite graph of enzyme-Target
* gpcr_KGE_drug.dat: Vector embedding of gpcr drugs (i.e. G-protein-coupled Receptors) 
* gpcr_KGE_target.dat: Vector embeddings of targets that interact with gpcr drugs.
* ic_KG.dat: Bipartite graph of Ion Channels-Target
* ic_KGE_drug.dat: Vector embedding of Ion Channels drugs
* ic_KGE_target.dat: Vector embeddings of targets that interact with Ion Channels drugs.
* ic_KG.dat: Bipartite graph of Ion Channels-Target

Note : It's worth noting that the folder 'data' contains all the similarity metrics, including Cosine similarity, Euclidean distance, Jaccard similarity, Manhattan distance, and Pearson correlation, depending on the selected strategies (i.e., contextual (KGE or ProtBERT) or local embedding (molecular fingerprint and protein characteristics) and the type of datasets (i.e., Enzymes, Ion Channels, G-protein-coupled Receptors).

# Choosing a classifier:
Choose the classifier to compute:
* ExtraTreesClassifier:
```
    model= ExtraTreesClassifier(n_estimators=trees, random_state=1357)
```
* DecisionTreeClassifier:
```
    model= DecisionTreeClassifier(random_state=1357)
```
* MLPClassifier:
```
    model= MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
```
* RandomForestClassifier:
```
    #model = RandomForestClassifier(n_estimators=trees, n_jobs=6, criterion=c, class_weight="balanced", random_state=1357)
```
# Choosing a given dataset:
Choose one from the GPCR, ENZ or IC datasets:

* gpcr_dataset:
```
    Mat_Int = np.loadtxt("gpcr_dataset/drug_target_association.txt", skiprows=1, usecols=range(1, 95), dtype=float)
    Mat_Sim_DD = np.loadtxt("gpcr_dataset/drug_drug_sim.txt", skiprows=1, usecols=range(1, 189), dtype=float)
    Mat_Sim_TT = np.loadtxt("gpcr_dataset/target_target_sim.txt", skiprows=1, usecols=range(1, 95), dtype=float)
    Name_T  = {line.strip(): i for i, line in enumerate(open('gpcr_dataset/target_list.txt')) if line.strip()}
    Name_D  = {line.strip(): i for i, line in enumerate(open('gpcr_dataset/drug_list.txt')) if line.strip()}
```
* enz_dataset:
```
    Mat_Int = np.loadtxt("enz_dataset/drug_target_association.txt", skiprows=1, usecols=range(1, 658), dtype=float)
    Mat_Sim_DD = np.loadtxt("enz_dataset/drug_drug_sim.txt", skiprows=1, usecols=range(1, 347), dtype=float)
    Mat_Sim_TT = np.loadtxt("enz_dataset/target_target_sim.txt", skiprows=1, usecols=range(1, 658), dtype=float)
    Name_T  = {line.strip(): i for i, line in enumerate(open('enz_dataset/target_list.txt')) if line.strip()}
    Name_D  = {line.strip(): i for i, line in enumerate(open('enz_dataset/drug_list.txt')) if line.strip()}
```
* ic_dataset:
```
    Mat_Int = np.loadtxt("ic_dataset/drug_target_association.txt", skiprows=1, usecols=range(1, 205), dtype=float)
    Mat_Sim_DD = np.loadtxt("ic_dataset/drug_drug_sim.txt", skiprows=1, usecols=range(1, 170), dtype=float)
    Mat_Sim_TT = np.loadtxt("ic_dataset/target_target_sim.txt", skiprows=1, usecols=range(1, 205), dtype=float)
    Name_T  = {line.strip(): i for i, line in enumerate(open('ic_dataset/target_list.txt')) if line.strip()}
    Name_D  = {line.strip(): i for i, line in enumerate(open('ic_dataset/drug_list.txt')) if line.strip()}
    ```
