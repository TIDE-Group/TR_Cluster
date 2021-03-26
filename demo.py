from time import time
from textrepresentation import TextRepresentation
from singlepass import ClusterUnit, ClusterUnitWVLDA, SinglePassCluster, TimedSinglePassCluster
from hac import HAC
import os
import numpy as np


doc_names = os.listdir('./doc/')
texts = []
for doc_name in doc_names:
    with open('./doc/' + doc_name, 'r', encoding='utf-8') as f:
        texts.append(f.read())

tr = TextRepresentation(texts, wv_dim=256, lda_n_components=256)

wv_tfidf = tr.wv_tfidf
lda_doc_topic = tr.lda_doc_topic

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

sp_cluster = SinglePassCluster(
    thresh=0.9,
    gamma=0.5,
    wv_tfidf=wv_tfidf,
    lda_doc_topic=lda_doc_topic
)

sp_cluster.print_result()

hac_cluster = HAC(sp_cluster.cluster_list, theta=0.8, gamma=0.5)

hac_cluster.print_result()

#加入时间窗口
time_sp_cluster = TimedSinglePassCluster(
    thresh=0.9,
    gamma=0.5,
    wv_tfidf=wv_tfidf,
    lda_doc_topic=lda_doc_topic,
    time_slices=np.arange(102).reshape(6, 17).tolist()
)

time_sp_cluster.print_result()

hac_cluster_ = HAC(time_sp_cluster.cluster_list, theta=0.8, gamma=0.5)

hac_cluster_.print_result()
pass