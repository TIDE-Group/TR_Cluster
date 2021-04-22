from time import time
from textrepresentation import TextRepresentation, BigBirdTextRepresentation, BERTTextRepresentation
from singlepass import SinglePassCluster, TimedSinglePassCluster
from utils import ClusterUnit, eval_clustering_results
from hac import HAC
import os
import numpy as np
import pandas as pd
import pickle

# doc_names = os.listdir('./doc/')
# texts = []
# for doc_name in doc_names:
#     with open('./doc/' + doc_name, 'r', encoding='utf-8') as f:
#         texts.append(f.read())
df = pd.read_excel('./cnn_final.xlsx')
df = df[df['content'].notna()]
df = df[df['label'].notna()]
df.index = range(len(df))
# print(df.columns)
texts = df['content'].tolist()
# labels = df[df['content'].notna()]['id'].tolist()
# week_set = df['date'].dt.week.unique()
# week_slices = [df[df['date'].dt.week == week]['id'].tolist() for week in week_set]
# week_set = df['week'].unique()
# week_slices = [df[df['week'] == week].index.tolist() for week in week_set]
time_indices = [df[df['date'] == date].index.tolist() for date in df['date'].sort_values(ascending=True).unique()]
labels = df['label'].astype('int').tolist()

# if os.path.exists('./pkls/tr.pkl'):
#     with open('./pkls/tr.pkl', 'rb') as f:
#         tr = pickle.load(f)
# else:
#     tr = TextRepresentation(texts, labels=labels, wv_dim=128, lda_n_components=256,  ner=False)
#     with open('./pkls/tr.pkl', 'wb') as f:
#         pickle.dump(tr, f)

if os.path.exists('./pkls/btr.pkl'):
    with open('./pkls/btr.pkl', 'rb') as f:
        btr = pickle.load(f)
else:
    btr = BERTTextRepresentation(texts, labels=labels, wv_dim=128, batch_size=10)
    with open('./pkls/btr.pkl', 'wb') as f:
        pickle.dump(btr, f)
    
# if os.path.exists('./pkls/bbtr.pkl'):
#     with open('./pkls/bbtr.pkl', 'rb') as f:
#         bbtr = pickle.load(f)
# else:
#     bbtr = BigBirdTextRepresentation(texts, labels=labels,  wv_dim=128, batch_size=10, ner=False)
#     with open('./pkls/bbtr.pkl', 'wb') as f:
#         pickle.dump(btr, f)



# wv_tfidf = tr.wv_tfidf
# lda_doc_topic = tr.lda_doc_topic

# # update
# tr1 = TextRepresentation()
# tr1.update(texts[:-1])
# wv_tfidf1 = tr1.wv_tfidf
# lda_doc_topic1 = tr1.lda_doc_topic
# tr1.update([texts[-1]])
# wv_tfidf2 = tr1.wv_tfidf
# lda_doc_topic2 = tr1.lda_doc_topic

# #sparse, 应付大数据

# tr2 = TextRepresentation(
#     texts, 
#     sparse=True,
#     wv_dim = 200,
#     lda_n_components = 200
# )

# wv_tfidf3 = tr2.wv_tfidf
# lda_doc_topic3 = tr2.lda_doc_topic

# sp_cluster = SinglePassCluster(
#     thresh=0.9,
#     gamma=0.5,
#     wv_tfidf=wv_tfidf,
#     lda_doc_topic=lda_doc_topic
# )

# sp_cluster.print_result()

# sp_cluster = SinglePassCluster(
#     clust_thresh=0.9,
#     text_representation=tr,
#     weight=np.array([0.5, 0.5, 0])
# )

# sp_cluster.print_result()

# hac_cluster = HAC(sp_cluster.cluster_list, clust_theta=0.9, weight = np.array([0.5, 0.5, 0]))
# hac_cluster.print_result()


# hac_cluster = HAC(sp_cluster.cluster_list, theta=0.8, gamma=0.5)

# hac_cluster.print_result()

# #加入时间窗口
# time_sp_cluster = TimedSinglePassCluster(
#     thresh=0.9,
#     gamma=0.5,
#     wv_tfidf=wv_tfidf,
#     lda_doc_topic=lda_doc_topic,
#     time_slices=np.arange(102).reshape(6, 17).tolist()
# )

time_sp_cluster = TimedSinglePassCluster(
    clust_thresh=0.95,
    text_representation=btr,
    weight=np.array([0.55, 0.45]),
    time_slices=time_indices
)
time_sp_cluster.print_result()

hac_cluster_ = HAC(time_sp_cluster.cluster_list, clust_theta=1.75, weight = np.array([0.55, 0.45]))
hac_cluster_.print_result()

clusting_results = [cluster.node_list for cluster in hac_cluster_.cluster_list]
acc, Pmacro, Rmacro, f1, pred_labels, labels = eval_clustering_results(btr._labels, clusting_results, ave=False)
print('acc: %.2f'%acc)
print('F1:', f1)

# print('acc: {:.2f}, P: {:.2f}, R: {:.2f}, F1: {:.2f}'.format(acc * 100 , Pmacro * 100 , Rmacro * 100 , f1 * 100 ))
# label_ = [[i for i in range(len(tr._labels)) if tr._labels[i] == j] for j in df['label'].unique()]
# print(label_)
# time_sp_cluster = TimedSinglePassCluster(
#     clust_thresh=0.98,
#     weight=[0.5, 0.5],
#     text_representation=bbtr,
#     time_slices=week_indice
# )

# time_sp_cluster.print_result()

# hac_cluster_ = HAC(time_sp_cluster.cluster_list, clust_theta=0.99, weight = [1, 0])
# hac_cluster_.print_result()

# hac_cluster_ = HAC(time_sp_cluster.cluster_list, theta=0.8, gamma=0.5)

# hac_cluster_.print_result()
pass