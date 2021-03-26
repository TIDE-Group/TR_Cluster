from sys import flags
import numpy as np
from numpy.lib.arraysetops import isin
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import time
from gensim import corpora
from singlepass import ClusterUnit, ClusterUnitWVLDA, SinglePassCluster, cos_sim_l
warnings.filterwarnings('ignore')

class HAC(object):
    """
    HAC层次聚类算法, 在SinglePass的基础上对已经聚类完毕的文本簇再进行一次聚类
    inputs:
    --------
    cluster_list: list[ClusterUnitWVLDA] / list[ClusterUnit], 已经初步聚类的簇列表
    theta: float, 聚合的余弦相似度阈值
    gamma: float, LDA向量和W2V&tfidf向量的融合权重
    """
    def __init__(self, cluster_list:list, theta:float = 0.9, gamma:float = 0.5):
        self.cluster_list = cluster_list
        self.theta = theta
        self.gamma = gamma
        if isinstance(cluster_list[0], ClusterUnit):
            t1 = time.time()
            self.doc_matrix = np.array([cluster.center for cluster in cluster_list]) # [num_doc, vec_dim]
            self.doc_hac()
            t2 = time.time()
            self.cluster_time = t2 - t1
        elif isinstance(cluster_list[0], ClusterUnitWVLDA):
            t1 = time.time()
            self.wv_matrix = np.array([cluster.center_wv for cluster in cluster_list]) # [num_doc, wv_dim]
            self.lda_matrix = np.array([cluster.center_lda for cluster in cluster_list]) # [num_doc, lda_n_components]
            self.lda_wv_hac()
            t2 = time.time()
            self.cluster_time = t2 - t1
        else:
            raise TypeError("elements of cluster_list not in class:[ClusterUnit, ClusterUnitWVLDA]")
    
    def lda_wv_hac(self):
        wv_matrix = self.wv_matrix
        lda_matrix = self.lda_matrix
        while(True):
            wv_sim = cosine_similarity(X=wv_matrix)
            lda_sim = cosine_similarity(X=lda_matrix)
            sim = self.gamma * wv_sim + (1 - self.gamma) * lda_sim #对角方阵 
            row, col = np.diag_indices_from(sim)
            sim[row, col] = np.array([0] * sim.shape[0])
            max_sim = np.max(sim)
            if max_sim < self.theta: #小于阈值, 聚类结束
                break
            else:
                idx = np.argmax(sim)
                row, col = idx // sim.shape[1], idx % sim.shape[1] #最大相似度的两篇文档在self.cluster_list中的索引
                cluster_1, cluster_2 = self.cluster_list[row], self.cluster_list[col]
                self.cluster_list.remove(cluster_1)
                self.cluster_list.remove(cluster_2)
                new_cluster = ClusterUnitWVLDA.union_cluster(cluster_1=cluster_1, cluster_2=cluster_2) #合并两个簇
                self.cluster_list.append(new_cluster)
                wv_matrix = np.array([cluster.center_wv for cluster in self.cluster_list])
                lda_matrix = np.array([cluster.center_lda for cluster in self.cluster_list])
    
    def doc_hac(self):
        doc_matrix = self.doc_matrix
        while(True):
            sim = cosine_similarity(X=doc_matrix)
            sim[row, col] = np.array([0] * sim.shape[0])
            max_sim = np.max(sim)
            if max_sim < self.theta: #小于阈值, 聚类结束
                break
            else:
                idx = np.argmax(sim)
                row, col = idx // sim.shape[1], idx % sim.shape[1] #最大相似度的两篇文档在self.cluster_list中的索引
                cluster_1, cluster_2 = self.cluster_list[row], self.cluster_list[col]
                self.cluster_list.remove(cluster_1)
                self.cluster_list.remove(cluster_2)
                new_cluster = ClusterUnit.union_cluster(cluster_1=cluster_1, cluster_2=cluster_2) #合并两个簇
                self.cluster_list.append(new_cluster)
                doc_matrix = np.array([cluster.center for cluster in self.cluster_list])
    
    def print_result(self):
        for idx, clusterunit in enumerate(self.cluster_list):
            print("cluster : %d"%idx)
            print(clusterunit.node_list)
        print('聚类数目: %d'%len(self.cluster_list))
        print('聚类耗时: %.6f s\n'%(self.cluster_time / 1000))
            

