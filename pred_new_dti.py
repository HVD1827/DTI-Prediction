#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys
import time
import numpy as np
import random
import os
import copy
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score, auc, precision_recall_fscore_support
from sklearn.metrics import roc_curve, accuracy_score, matthews_corrcoef, f1_score, confusion_matrix

from Adding_identifier_to_embedding import adding_identifier_to_identifier
from embedding_similarities import embedding_similarities
from local_embedding.graph_to_matrix import convert_drug_drug_similarities_to_matrix, \
    convert_target_target_similarities_to_matrix


def main():
    parser = ArgumentParser("KGE_new_DTI",
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--train-data',
                        default=r'../rating_train.dat',
                        help='Input bipartite DTI file.')

    parser.add_argument('--KGE-drug',
                        default=r'../KGE_drug.dat',
                        help="file of embedding vectors of drugs")

    parser.add_argument('--KGE-target',
                        default=r'../KGE_target.dat',
                        help="file of embedding vectors of targets")

    parser.add_argument('--type', default='SP', type=str,
                        help='types of DTI prediction')

    parser.add_argument('--fold', default='20', type=str,
                        help='number of CV folds')

    parser.add_argument('--cycle', default='1', type=str,
                        help='number of excecutions of 20-fold CV')

    parser.add_argument('--trees', default=1, type=int,
                        help='number of trees')

    parser.add_argument('--c', default='gini', type=str,
                        help='parameter of split')

    parser.add_argument('--concat', default=3, type=int,
                        help='scheme of the drug-target embedding generation')

    # data: e/gpcr-euc/ic, we are using IC dataset 
    data = 'ic'
    base_root = './data/ic-KGE/{}_dataset'.format(data)
    args = parser.parse_args(['--train-data', os.path.join(base_root, '{}_KG.dat'.format(data)),
                              '--KGE-drug', os.path.join(base_root, '{}_KGE_drug.dat'.format(data)),
                              '--KGE-target', os.path.join(base_root, '{}_KGE_target.dat'.format(data)),
                              ])

    predict(args)


def predict(args):
    # gpcr-euc_dataset:
    Mat_Int = np.loadtxt("./data/ic-KGE/ic_dataset/matbipc.txt", skiprows=1,
                         usecols=range(1, 205), dtype=float)
    Mat_Sim_DD = np.loadtxt("./data/ic-KGE/ic_dataset/drug.txt", skiprows=1,
                            usecols=range(1, 170), dtype=float)
    Mat_Sim_TT = np.loadtxt("./data/ic-KGE/ic_dataset/target.txt", skiprows=1,
                            usecols=range(1, 205), dtype=float)
    Name_T = {line.strip(): i for i, line in
              enumerate(open('./data/ic-KGE/ic_dataset/fichier_sortie.txt')) if line.strip()}
    Name_D = {line.strip(): i for i, line in
              enumerate(open('./data/ic-KGE/ic_dataset/fichier_colonne.txt')) if line.strip()}

    # New_DTI = Drug_Target_Interaction_Prediction(args=args, Mat_Int=Mat_Int, Mat_Sim_DD=Mat_Sim_DD,
                                                #  Mat_Sim_TT=Mat_Sim_TT, Name_D=Name_D, Name_T=Name_T, threshold=0.001)
    New_BE = BE_Prediction(args=args, Mat_Int=Mat_Int, Mat_Sim_DD=Mat_Sim_DD,
                           Mat_Sim_TT=Mat_Sim_TT, Name_D=Name_D, Name_T=Name_T)
    New_DTI = 0

    return (New_DTI, New_BE)

def BE_Prediction(args, Mat_Int, Mat_Sim_DD, Mat_Sim_TT, Name_D, Name_T):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    # Load be.csv
    be_df = pd.read_csv("be.csv")  # columns: drug,target,binding_energy
    be_dict = {(r["drug"], r["target"]): r["binding_energy"] for _, r in be_df.iterrows()}

    # Load embeddings
    print("Loading embeddings for BE")
    emb_D = pd.read_csv(args.KGE_drug, sep=' ', index_col=0, header=None)
    emb_T = pd.read_csv(args.KGE_target, sep=' ', index_col=0, header=None)
    # print first entry of embd and t
    print(emb_D.head(1))
    print(emb_T.head(1))
    print("First drugs in embedding:", list(emb_D.index)[:5])
    print("First targets in embedding:", list(emb_T.index)[:5])
    print("First drug IDs in Name_D:", list(Name_D.keys())[:5])
    print("First target IDs in Name_T:", list(Name_T.keys())[:5])

    data = []
    for (drug, target), be in be_dict.items():
        try:
            print('Processing drug:', drug, 'target:', target)
            V_emb_drug = np.array(emb_D.loc['u'+drug][:-1])
            V_emb_target = np.array(emb_T.loc['i'+target][:-1])
            # DEBUG PRINTS
            # print('V_emb_drug:', V_emb_drug)
            # print('V_emb_target:', V_emb_target)
            # print('Processing drug:', drug, 'target:', target)
            # DEBUG PRINTS
            num_row = Name_D[drug[1:] if drug.startswith('u') else drug]
            num_col = Name_T[target[1:] if target.startswith('i') else target]
            print(f"Processing ({drug}, {target}) with BE: {be}")
            Drug_fusion = Vector_D_fusion(num_row=num_row, num_col=num_col,
                                          V_emb_drug=V_emb_drug,
                                          v_emb_target=V_emb_target,
                                          Mat_Sim_DD=Mat_Sim_DD,
                                          Mat_Int=Mat_Int,
                                          sort_concat=args.concat)

            Target_fusion = Vector_T_fusion(num_row=num_row, num_col=num_col,
                                            V_emb_drug=V_emb_drug,
                                            v_emb_target=V_emb_target,
                                            Mat_Sim_TT=Mat_Sim_TT,
                                            Mat_Int=Mat_Int,
                                            sort_concat=args.concat)

            x = np.concatenate((np.array(Drug_fusion), np.array(Target_fusion)))
            data.append((x, be))
        except Exception as e:
            print(f"Skipped pair ({drug}, {target}):", e)

    if not data:
        print("No usable data for binding energy regression.")
        return None

    X, y = zip(*data)
    X, y = np.array(X), np.array(y)
    # note: 20 is just a seed for random number generation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    # different models
    model = RandomForestRegressor(n_estimators=100, random_state=20)
    from sklearn.linear_model import LinearRegression
    # model = LinearRegression()
    # model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=20)
    # fit.............................................................................
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end_time = time.time()
    print("Time taken for training and prediction:", end_time - start_time)
    print("Binding Energy Prediction Results:")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))

    return model


def Drug_Target_Interaction_Prediction(args, Mat_Int, Mat_Sim_DD, Mat_Sim_TT, Name_D, Name_T, threshold=0.001):
    # data loading and initialisation
    Gr_Bi = pd.read_csv(args.train_data, sep='\t', header=None)
    Vector_emb_D = args.KGE_drug
    Vector_emb_T = args.KGE_target
    row, col = Gr_Bi.shape[0], Gr_Bi.shape[1]
    # drug-tgt maps
    D, T = {}, {}
    Drug_name, Target_name = set(), set()
    Full_Drug, Full_Target = {}, {}
    for i in range(row):
        if (Gr_Bi[0].iloc[i] not in D.keys()):
            D[Gr_Bi[0].iloc[i]] = []
        D[Gr_Bi[0].iloc[i]].append(Gr_Bi[1].iloc[i])
        Drug_name.update([Gr_Bi[0].iloc[i]])
    for i in range(row):
        if (Gr_Bi[1].iloc[i] not in T.keys()):
            T[Gr_Bi[1].iloc[i]] = []
        T[Gr_Bi[1].iloc[i]].append(Gr_Bi[0].iloc[i])
        Target_name.update([Gr_Bi[1].iloc[i]])
    # generation of negative samples
    NBR_DTI = len(Drug_name) * len(Target_name)
    print('total number of drug-target pairs in the DTIs space:', NBR_DTI)
    for i in list(Drug_name):
        if (i not in Full_Drug):
            Full_Drug[i] = []
        for j in list(Target_name):
            Full_Drug[i].append(j)
    for i in list(Target_name):
        if (i not in Full_Target):
            Full_Target[i] = []
        for j in list(Drug_name):
            Full_Target[i].append(j)
    #### generate negative sample/ those who don't interact
    if (args.type == 'SP'):
        count = 0
        for i in Drug_name:
            Element = Full_Drug[i]
            for j in D[i]:
                try:
                    Element.remove(j)
                    count += 1
                except ValueError:
                    pass
    # feature generation and crs validn
    AUPR_total, AUC_total = [], []

    for i in range(int(args.cycle)):
        if (args.type == 'SP'):
            print('=============== Processing the SP task ==================')
            New_v_couple_DTI = create_new_vector(Vector_emb_D=Vector_emb_D, Vector_emb_T=Vector_emb_T,
                                                 vertice_total=Full_Drug, D=D, Mat_Int=Mat_Int, Mat_Sim_DD=Mat_Sim_DD,
                                                 Mat_Sim_TT=Mat_Sim_TT, Name_D=Name_D, Name_T=Name_T,
                                                 fold_nums=args.fold,
                                                 sort_concat=args.concat)
            all_scores, all_AUPR, all_AUC, name_DTI = Cross_validation(New_v_couple_DTI=New_v_couple_DTI,
                                                                       trees=args.trees, c=args.c,
                                                                       fold_nums=args.fold)
            possible_new_dti = []
            fold_nums = int(args.fold)
            threshold = float(threshold)
            for temp_ in range(fold_nums):
                cal = dict(zip(name_DTI[temp_], all_scores[temp_]))
                cal_ = []
                for i in cal:
                    if (cal[i] > threshold):
                        cal_.append(i)
                for i in D:
                    for j in D[i]:
                        if (('u' + i, 'i' + j) in cal_):
                            cal_.remove(('u' + i, 'i' + j))
                for i in cal_:
                    possible_new_dti.append([(i[0][2:], i[1][2:]), cal[i]])
            possible_new_dti.sort()
            matrix_ = pd.DataFrame(possible_new_dti)
            if not (matrix_.empty):
                matrix_ = matrix_.sort_values(1, ascending=False)
                New_DTI = np.array(matrix_)[:, :1][:30]
                print('===== Predicted DTIs with top 30 probability scores =====')
                print(New_DTI)
            else:
                New_DTI = np.array(0)
                print('================= No DTIs are predicted =================')

        AUPR_total.append(np.mean(all_AUPR))
        AUC_total.append((np.mean(all_AUC)))
    return New_DTI


def Vector_D_fusion(num_row, num_col, V_emb_drug, v_emb_target, Mat_Sim_DD, Mat_Int, sort_concat=1):
    row1, col1 = Mat_Sim_DD.shape
    row2, col2 = Mat_Int.shape
    sim = []
    count = 0
    for Nb in range(row2):
        if (Nb != num_row):
            if (sort_concat == 1):
                Vp = np.dot(np.dot(V_emb_drug, Mat_Sim_DD[num_row][Nb]), Mat_Int[Nb][num_col])
                Vp = np.multiply(Vp, v_emb_target)
            if (sort_concat == 2):
                work1 = np.dot(V_emb_drug, Mat_Sim_DD[num_row][Nb])
                work2 = np.dot(v_emb_target, Mat_Int[Nb][num_col])
                Vp = np.concatenate((work1, work2))
            if (sort_concat == 3):
                work1 = np.dot(V_emb_drug, Mat_Sim_DD[num_row][Nb])
                work2 = np.dot(v_emb_target, Mat_Int[Nb][num_col])
                Vp = np.array(work1 + work2)
            count += Vp

    return count


def Vector_T_fusion(num_row, num_col, V_emb_drug, v_emb_target, Mat_Sim_TT, Mat_Int, sort_concat=1):
    row1, col1 = Mat_Int.shape
    row2, col2 = Mat_Sim_TT.shape
    sim = []
    count = 0
    for Nb in range(row2):
        if (Nb != num_col):
            if (sort_concat == 1):
                Vp = np.dot(np.dot(V_emb_drug, Mat_Int[num_row][Nb]), Mat_Sim_TT[Nb][num_col])
                Vp = np.multiply(Vp, v_emb_target)
            if (sort_concat == 2):
                work1 = np.dot(V_emb_drug, Mat_Int[num_row][Nb])
                work2 = np.dot(v_emb_target, Mat_Sim_TT[Nb][num_col])
                Vp = np.concatenate((work1, work2))
            if (sort_concat == 3):
                work1 = np.dot(V_emb_drug, Mat_Int[num_row][Nb])
                work2 = np.dot(v_emb_target, Mat_Sim_TT[Nb][num_col])
                Vp = np.array(work1 + work2)
            count += Vp

    return count


def Cross_validation(New_v_couple_DTI, trees, c, fold_nums=20):
    all_scores, all_AUPR, all_AUC, all_accuracy, all_mcc, all_f1, all_conf_matrix, name_DTI = [], [], [], [], [], [], [], []
    all_time = []
    fold_nums = int(fold_nums)
    counter = 0
    for fold_num in range(fold_nums):
        counter += 1
        index = [fold_num]
        # index_remain=set(list_)-set(index)
        temp_test = New_v_couple_DTI[fold_num]
        temp_train = copy.deepcopy(New_v_couple_DTI)
        temp_train.pop(fold_num)
        random.shuffle(temp_test)
        random.shuffle(temp_train)
        X_train, Y_train, train_pair_name = [], [], []
        print(len(temp_train), len(temp_test))
        # len(temp_train)=9,len(temp_test)=32182
        for one_fold in temp_train:
            for i in one_fold:
                X_train.append(i[0])
                Y_train.append(i[1])
                train_pair_name.append((i[2], i[3]))
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test, Y_test, test_pair_name = [], [], []
        for i in temp_test:
            # print(i[1])
            # if(i[1]==0):
            X_test.append(i[0])
            Y_test.append(i[1])
            test_pair_name.append((i[2], i[3]))
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        print('Start the {}th fold novel DTI prediction'.format(counter))
        start_time = time.time()
        scores_testing, AUPR, AUC, test_pair_name, accuracy, mcc, f1, conf_matrix = classifiers(X_train, Y_train,
                                                                                                X_test, Y_test,
                                                                                                test_pair_name,
                                                                                                trees=trees, c=c,
                                                                                                )
        end_time = time.time()
        all_time.append(end_time - start_time)
        all_scores.append(scores_testing)
        print('Fold{} AUPR:{:.5f}, AUC:{:.5f}, Accuracy:{:.5f}, MCC:{:.5f}, F1:{:.5f}'.format(fold_num, AUPR, AUC,
                                                                                              accuracy, mcc, f1))
        all_AUPR.append(AUPR)
        all_AUC.append(AUC)
        all_accuracy.append(accuracy)
        all_mcc.append(mcc)
        all_f1.append(f1)
        all_conf_matrix.append(conf_matrix)
        name_DTI.append(test_pair_name)

    mean_AUPR = np.mean(all_AUPR)
    mean_AUC = np.mean(all_AUC)
    mean_accuracy = np.mean(all_accuracy)
    mean_mcc = np.mean(all_mcc)
    mean_f1 = np.mean(all_f1)
    total_time = np.mean(all_time)*fold_nums
    print('mean_AUPR:{:.5f}, mean_AUC:{:.5f}, mean_Accuracy:{:.5f}, mean_MCC:{:.5f}, mean_F1:{:.5f}, total_time:{:.5f}'.format(mean_AUPR,
                                                                                                            mean_AUC,
                                                                                                            mean_accuracy,
                                                                                                            mean_mcc,
                                                                                                            mean_f1,
                                                                                                            total_time))

    # Vous pouvez également imprimer ou enregistrer les matrices de confusion pour chaque fold ici
    for i, conf_matrix in enumerate(all_conf_matrix):
        print('Fold{} Confusion Matrix:'.format(i))
        print(conf_matrix)

    return all_scores, all_AUPR, all_AUC, name_DTI


def classifiers(X_train, Y_train, X_test, Y_test, test_pair_name, trees, c):
    max_abs_scaler = MaxAbsScaler()
    X_train_maxabs_fit = max_abs_scaler.fit(X_train)
    X_train_maxabs_transform = max_abs_scaler.transform(X_train)
    X_test_maxabs_transform = max_abs_scaler.transform(X_test)

    # model= ExtraTreesClassifier(n_estimators=trees, random_state=1357)
    # model= DecisionTreeClassifier(random_state=1357)
    # model= MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)
    # model = RandomForestClassifier(n_estimators=trees, n_jobs=6, criterion=c, class_weight="balanced", random_state=1357)
    # model = svm.SVC(decision_function_shape='ovo')
    # model = SGDClassifier(loss="log", penalty="l2", max_iter=5)
    # model = NearestCentroid()
    # model = GaussianNB()
    # model = HistGradientBoostingClassifier(max_iter=100).fit(X_train, y_train)
    # model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    # model = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
    # model = SVC(kernel='rbf', probability=True)

    ############-----models------#######################
    # model = KNeighborsClassifier(n_neighbors=2)
    # model = svm.SVC(decision_function_shape='ovo') useless
    # model = NearestCentroid()
    from sklearn.ensemble import HistGradientBoostingClassifier
    # model = HistGradientBoostingClassifier(max_iter=100).fit(X_train, Y_train)
    # model = KNeighborsClassifier(n_neighbors=8)
    from xgboost import XGBClassifier
    # model = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    from lightgbm import LGBMClassifier
    # model = LGBMClassifier(n_estimators=20, learning_rate=0.1, random_state=42)
    from catboost import CatBoostClassifier
    # model = CatBoostClassifier(iterations=20, learning_rate=0.1, depth=6, verbose=0)
    #include ridge
    from sklearn.linear_model import RidgeClassifier
    # model = RidgeClassifier(alpha=1.0)
    from sklearn.linear_model import PassiveAggressiveClassifier
    # model = PassiveAggressiveClassifier(max_iter=500, random_state=42)
    from sklearn.linear_model import Perceptron
    # model = Perceptron(tol=1e-3, random_state=42)
    from sklearn.ensemble import VotingClassifier, StackingClassifier
#     model = VotingClassifier(estimators=[
#     ('rf', RandomForestClassifier(n_estimators=100)),
#     ('svc', SVC(probability=True)),
#     ('mlp', MLPClassifier(hidden_layer_sizes=(5, 2), solver='adam'))
# ], voting='soft')
    model = StackingClassifier(estimators=[
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('svc', SVC(probability=True))
], final_estimator=LogisticRegression())
#     model = ComplementNB(alpha=1.0)
#     model = BernoulliNB(alpha=1.0)
#     model RadiusNeighborsClassifier(radius=1.0)
#     model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), solver='adam', max_iter=200)
#     model = Sequential([
#     Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
#     MaxPooling2D(pool_size=(2,2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax')
# ])
    from sklearn.linear_model import LogisticRegression
    # model = LogisticRegression(
    #     solver='saga',       # Faster stochastic solver for large datasets
    #     penalty='l2',        # Default regularization
    #     tol=0.1,             # Looser convergence tolerance
    #     max_iter=50,        # Reduced iterations
    #     n_jobs=-1,           # Use all cores
    #     random_state=1357,
    #     C=1.0                # Regularization strength
    # )
    ############-----end models----#######################
    model.fit(X_train_maxabs_transform, Y_train)
    test_prob = model._predict_proba_lr(X_test_maxabs_transform)[:, 1]
    precision, recall, _ = precision_recall_curve(Y_test, test_prob)

    AUPR = auc(recall, precision)
    AUC = roc_auc_score(Y_test, test_prob)

    # Ajouter la prédiction de classe (0 ou 1)
    test_predictions = (test_prob >= 0.5).astype(int)

    # Calculer les métriques d'évaluation
    accuracy = accuracy_score(Y_test, test_predictions)
    mcc = matthews_corrcoef(Y_test, test_predictions)
    f1 = f1_score(Y_test, test_predictions)

    # Matrice de confusion
    conf_matrix = confusion_matrix(Y_test, test_predictions)

    return test_prob, AUPR, AUC, test_pair_name, accuracy, mcc, f1, conf_matrix


def create_new_vector(Vector_emb_D, Vector_emb_T, vertice_total, D, Mat_Int, Mat_Sim_DD, Mat_Sim_TT, Name_D, Name_T,
                      fold_nums=20,
                      sort_concat=1):
    Vector_emb_D = pd.read_csv(Vector_emb_D, sep=' ', index_col=0, header=None)
    Vector_emb_T = pd.read_csv(Vector_emb_T, sep=' ', index_col=0, header=None)
    negative = []

    for j in vertice_total:
        for k in vertice_total[j]:
            negative.append((j, k))
    random.shuffle(negative)

    # this divides the set of negative samples into 20 folds, each used as test set for one fold
    testing = np.array_split(negative, int(fold_nums))
    testing = [list(fold) for fold in testing]

    pos_neg = []
    for j in testing:
        temp = []
        for k in j:
            temp.append(tuple(list(k) + [0]))

        for l in D:
            for m in D[l]:
                temp.append((l, m, 1))

        random.shuffle(temp)
        pos_neg.append(temp)

    New_v_couple_DTI = []
    counter = 0
    for i in pos_neg:
        counter += 1
        print('Start the {}th fold embedding generation'.format(counter))
        temp = []
        for j in i:
            V_emb_drug = np.array(Vector_emb_D.loc[j[0]][:-1])
            v_emb_target = np.array(Vector_emb_T.loc[j[1]][:-1])
            num_row = Name_D[j[0][1:]]
            num_col = Name_T[j[1][1:]]

            Drug_fusion = Vector_D_fusion(num_row=num_row, num_col=num_col, V_emb_drug=V_emb_drug,
                                          v_emb_target=v_emb_target,
                                          Mat_Sim_DD=Mat_Sim_DD,
                                          Mat_Int=Mat_Int,
                                          sort_concat=sort_concat)
            Target_fusion = Vector_T_fusion(num_row=num_row, num_col=num_col, V_emb_drug=V_emb_drug,
                                            v_emb_target=v_emb_target,
                                            Mat_Sim_TT=Mat_Sim_TT,
                                            Mat_Int=Mat_Int,
                                            sort_concat=sort_concat)
            x = np.concatenate((np.array(Drug_fusion), np.array(Target_fusion)))
            y = j[2]
            add_u = 'u' + j[0]
            add_i = 'i' + j[1]
            temp.append([x, y, add_u, add_i])
        New_v_couple_DTI.append(temp)

    return New_v_couple_DTI


if __name__ == "__main__":
    embedding_similarities()
    convert_drug_drug_similarities_to_matrix("./data/input_embedding_dict/ic/eucludian_similarities_ic_proteins.txt")
    convert_target_target_similarities_to_matrix("./data/input_embedding_dict/ic/eucludian_similarities_ic_proteins.txt")
    adding_identifier_to_identifier()
    # preprocessing done, now call main function
    sys.exit(main())