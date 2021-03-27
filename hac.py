from typing import List, Union
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import time
from gensim import corpora
from singlepass import ClusterUnit
import typesentry
tc1 = typesentry.Config()
Isinstance = tc1.is_type
warnings.filterwarnings('ignore')

class HAC(object):
    """
    HAC层次聚类算法, 在SinglePass的基础上对已经聚类完毕的文本簇再进行一次聚类
    
    inputs:
    --------
    cluster_list: List[ClusterUnit], 已经初步聚类的簇列表
    clust_theta: float, 聚合的余弦相似度阈值
    weight: Union[float, List[float]], text_representation中各种表示方法的权重

    弃用:
        gamma: float, LDA向量和W2V&tfidf向量的融合权重
    """
    def __init__(self, cluster_list:List[ClusterUnit], clust_theta:float = 0.9, weight: Union[float, List[float]]= 0.75):
        self.cluster_list = cluster_list
        self.clust_theta = clust_theta
        self.feature = cluster_list[0].feature

        if Isinstance(weight, float):
            if self.feature == 2:
                self.weight = np.array([weight, 1 - weight]) # 1, 2项权重
            else:
                raise RuntimeError("目前不支持三个以上的特征矩阵进行HAC")

        elif Isinstance(weight, List[float]) or Isinstance(weight, np.ndarray):
            assert np.sum(weight) == 1
            self.weight = np.array(weight)
            assert self.weight.ndim == 1
            assert self.feature == (len(self.weight))
        else:
            raise TypeError("weight 参数类型错误")
        
        """
        if Isinstance(cluster_list[0], ClusterUnit):
            t1 = time.time()
            self.doc_matrix = np.array([cluster.center for cluster in cluster_list]) # [num_doc, vec_dim]
            self.doc_hac()
            t2 = time.time()
            self.cluster_time = t2 - t1
        elif Isinstance(cluster_list[0], ClusterUnitWVLDA):
            t1 = time.time()
            self.wv_matrix = np.array([cluster.center_wv for cluster in cluster_list]) # [num_doc, wv_dim]
            self.lda_matrix = np.array([cluster.center_lda for cluster in cluster_list]) # [num_doc, lda_n_components]
            self.lda_wv_hac()
            t2 = time.time()
            self.cluster_time = t2 - t1
        else:
            raise TypeError("elements of cluster_list not in class:[ClusterUnit, ClusterUnitWVLDA]")
        """
        t1 = time.time()
        self._hac()
        t2 = time.time()
        self.cluster_time = t2 - t1
        


    """
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
    """
    def _hac(self):
        feature_matrixs = [np.concatenate([cluster.centers[i][None, :] for cluster in self.cluster_list], axis=0) \
            for i in range(self.feature)] #[feature, [n_topic, feature_dim]]
        
        while(True):
            sims = [cosine_similarity(X=feature_matrix) for feature_matrix in feature_matrixs] # [feature, [n_topic, n_topic]]
            sim = np.zeros_like(sims[0]) #对角方阵 [n_topic, n_topic]
            for i in range(self.feature):
                sim += self.weight[i] * sims[i]

            row, col = np.diag_indices_from(sim)
            sim[row, col] = np.array([0] * sim.shape[0])

            max_sim = np.max(sim)
            if max_sim < self.clust_theta: #小于阈值, 聚类结束
                break
            
            else:
                idx = np.argmax(sim)
                row, col = idx // sim.shape[1], idx % sim.shape[1] #最大相似度的两个话题在self.cluster_list中的索引
                cluster_1, cluster_2 = self.cluster_list[row], self.cluster_list[col]
                self.cluster_list.remove(cluster_1)
                self.cluster_list.remove(cluster_2)
                new_cluster = ClusterUnit.union_cluster(cluster_1=cluster_1, cluster_2=cluster_2) #合并两个簇
                self.cluster_list.append(new_cluster)

                feature_matrixs = [np.concatenate([cluster.centers[i][None, :] for cluster in self.cluster_list], axis=0) \
                    for i in range(self.feature)] #[feature, [n_topic, feature_dim]]


    
    def print_result(self):
        for idx, clusterunit in enumerate(self.cluster_list):
            print("cluster : %d"%idx)
            print(clusterunit.node_list)
        print('聚类话题数目: %d'%len(self.cluster_list))
        print('聚类耗时: %.6f s\n'%(self.cluster_time))
            

