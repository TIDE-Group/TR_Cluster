from typing import List, Union, Set
from textrepresentation import TextRepresentation, BigBirdTextRepresentation, BERTTextRepresentation
import numpy as np
import scipy.sparse as sp
import warnings
import time
from gensim import corpora
from utils import ClusterUnit
import typesentry
tc1 = typesentry.Config()
Isinstance = tc1.is_type
warnings.filterwarnings('ignore')

"""
class ClusterUnit:
    def __init__(self):
        self.node_list = []
        self.center = None
    
    def add_node(self, node_id:int, node_vec:np.ndarray):
        self.node_list.append(node_id)

        try:
            self.center = ((len(self.node_list) - 1) * self.center + node_vec) / len(self.node_list)
        except TypeError:
            self.center = node_vec # 初始化质心
    
    
    def add_node_sparse(self, node_id:int, node_vec: sp.csr_matrix):
        self.node_list.append(node_id)

        try:
            self.center = ((len(self.node_list) - 1) * self.center + node_vec) / len(self.node_list)
        except TypeError:
            self.center = node_vec # 初始化质心
    
    
    def remove_node(self, node_id:int, node_vec:np.ndarray):
        try:
            self.center = (len(self.node_list) * self.center - node_vec) / (len(self.node_list) - 1)
            self.node_list.remove(node_id)
        except ValueError:
            raise ValueError("%d 不在这个聚类中"%node_id)
    
    def move_node(self, 
        node_id:int, node_vec: np.ndarray,
        moved_cluster
    ): #将本簇的结点移到另一个簇
        self.remove_node(node_id)
        moved_cluster.add_node(node_id, node_vec)
    
    def add_cluster(
        self,
        added_cluster
    ): #将本簇完全添加到另一个簇
        try:
            added_cluster.center = (len(added_cluster.node_list) * added_cluster.center + len(self.node_list) * self.center) / \
                (len(added_cluster.node_list) + len(self.node_list))
            
        except TypeError:
            added_cluster.center = self.center
        added_cluster.node_list.extend(self.node_list)
    
    @staticmethod
    def union_cluster(cluster_1, cluster_2): #静态方法，合并两个簇
        assert len(cluster_1) > 0
        assert len(cluster_2) > 0
        union = ClusterUnit()
        union.node_list = cluster_1.node_list
        union.node_list.extend(cluster_2.node_list)
        union.center = (len(cluster_1) * cluster_1.center + len(cluster_2) * cluster_2.center) / (len(cluster_1) + len(cluster_2))
        return union

class ClusterUnit2F:
    def __init__(self):
        self.node_list = []
        self.center_1 = None
        self.center_2 = None
    
    def __len__(self):
        return len(self.node_list)
    
    def add_node(self, node_id:int, vec_1:np.ndarray, vec_2:np.ndarray):
        self.node_list.append(node_id)
        try:
            # self.center = ((len(self.node_list) - 1) * self.center + node_vec) / len(self.node_list)
            self.center_1 = ((len(self.node_list) - 1) * self.center_1 + vec_1) / len(self.node_list)
            self.center_2 = ((len(self.node_list) - 1) * self.center_2 + vec_2) / len(self.node_list)
        except TypeError:
            # self.center = node_vec # 初始化质心
            self.center_1 = vec_1
            self.center_2 = vec_2
    
    def remove_node(self, node_id:int, vec_1:np.ndarray, vec_2:np.ndarray):
        try:
            self.center_1 = (len(self.node_list) * self.center_1 - vec_1) / (len(self.node_list) - 1)
            self.center_2 = (len(self.node_list) * self.center_2 - vec_2) / (len(self.node_list) - 1)
            self.node_list.remove(node_id)
        except ValueError:
            raise ValueError("%d 不在这个聚类中"%node_id)
    
    def move_node(self, 
        node_id:int, vec_1: np.ndarray, vec_2: np.ndarray, 
        moved_cluster
    ): #将本簇的结点移到另一个簇
        self.remove_node(node_id)
        moved_cluster.add_node(node_id, vec_1, vec_2)
    
    def add_cluster(
        self,
        added_cluster
    ): #将本簇完全添加到另一个簇
        try:
            added_cluster.center_1 = (len(added_cluster.node_list) * added_cluster.center_1 + len(self.node_list) * self.center_1) / \
                (len(added_cluster.node_list) + len(self.node_list))
            added_cluster.center_2 = (len(added_cluster.node_list) * added_cluster.center_2 + len(self.node_list) * self.center_2) / \
                (len(added_cluster.node_list) + len(self.node_list))
        except TypeError:
            added_cluster.center_1 = self.center_1
            added_cluster.center_2 = self.center_2
        # for node_id in self.node_list:
        #     added_cluster.node_list.append(node_id)
        added_cluster.node_list.extend(self.node_list)
    
    @staticmethod
    def union_cluster(cluster_1, cluster_2): #静态方法，合并两个簇
        assert len(cluster_1) > 0
        assert len(cluster_2) > 0
        union = ClusterUnit2F()
        union.node_list = cluster_1.node_list
        union.node_list.extend(cluster_2.node_list)
        union.center_1 = (len(cluster_1) * cluster_1.center_1 + len(cluster_2) * cluster_2.center_1) / (len(cluster_1) + len(cluster_2))
        union.center_2 = (len(cluster_1) * cluster_1.center_2 + len(cluster_2) * cluster_2.center_2) / (len(cluster_1) + len(cluster_2))
        
        return union
"""



def cos_sim(v_a : np.ndarray, v_b : np.ndarray):
    cos = np.vdot(v_a, v_b) / (np.linalg.norm(v_a, 2) * np.linalg(v_b, 2))
    return cos

cos_sim_l = lambda v_a, v_b: np.vdot(v_a, v_b) / (np.linalg.norm(v_a, 2) * np.linalg.norm(v_b, 2))

class SinglePassCluster:
    """
    Single-Pass聚类算法
    
    inputs:
    --------
    clust_thresh: float, 相似度阈值
    weight: Union[float, List[float]], text_representation中各种表示方法的权重
    cluster_list: List[ClusterUnit], 预定义的聚合簇
    text_representation: Union[TextRepresentation, BigBirdTextRepresentation, BERTTextRepresentation], 文本表示的实例

    
    弃用:
    --------
        gamma: float, LDA向量和W2V&tfidf向量的融合权重
        vector_mat: np.ndarray, 输入的只有单个特征矩阵, 不可与wv_tfidf、lda_doc_topic共存
        wv_tfidf: np.ndarray, [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量
        lda_doc_topic: np.ndarray, [n_doc, lda_n_components]的np.ndarray, LDA训练的文档-主题矩阵
    """
    def __init__(
        self, 
        clust_thresh: float = 0.9, 
        weight: Union[float, List[float], np.ndarray] = 0.75,
        text_representation: Union[TextRepresentation, BigBirdTextRepresentation, BERTTextRepresentation] = None,
        cluster_list: List[ClusterUnit] = None    
    ):
        self.clust_thresh = clust_thresh
        if Isinstance(weight, float):
            self.weight = np.array([weight, 1 - weight], 0) # 1, 2项权重
        elif Isinstance(weight, List[float]) or Isinstance(weight, np.ndarray):
            self.weight = np.array(weight)
        else:
            raise TypeError("weight 参数类型错误")
        
        if cluster_list is not None:
            self.cluster_list = cluster_list
        else:
            self.cluster_list = []  # 聚类后簇的列表
        
        if Isinstance(text_representation, TextRepresentation):
            assert text_representation.wv_tfidf.shape[0] == text_representation.lda_doc_topic.shape[0]
            self.doc_num = text_representation.wv_tfidf.shape[0]
            self.feature_matrixs = [text_representation.wv_tfidf, text_representation.lda_doc_topic] # 重要性降序
            self.feature = 2 # 特征矩阵数目
            
        elif Isinstance(text_representation, BigBirdTextRepresentation):
            assert text_representation.big_bird_rp.shape[0] == text_representation.wv_tfidf.shape[0]
            self.doc_num = text_representation.wv_tfidf.shape[0]
            self.feature_matrixs = [text_representation.big_bird_rp, text_representation.wv_tfidf] # 重要性降序
            self.feature = 2 # 特征矩阵数目
        elif Isinstance(text_representation, BERTTextRepresentation):
            assert text_representation.bert_rp.shape[0] == text_representation.wv_tfidf.shape[0]
            self.doc_num = text_representation.wv_tfidf.shape[0]
            self.feature_matrixs = [text_representation.bert_rp, text_representation.wv_tfidf]
            self.feature = 2
        else:
            raise TypeError("text_representation 类型错误")
        self.entities = text_representation.entities
        self.ids = text_representation.ids
        assert len(self.ids) == self.doc_num
        t1 = time.time()
        self.clustering()
        t2 = time.time()
        
        self.spend_time = t2 - t1  # 聚类花费的时间
        self.cluster_num = len(self.cluster_list) # 聚类完成后 簇的个数
    
    """
    def clustering(self, WVLDA=True):
        if WVLDA:
            if self.cluster_list == []:
                init_node = ClusterUnit2F()
                init_node.add_node(node_id=0, wv_vec=self.wv_tfidf[0, :], lda_vec=self.lda_doc_topic[0, :])
                self.cluster_list.append(init_node)
            for idx in range(1, self.doc_num):
                sim = np.zeros(len(self.cluster_list))
                for i in range(len(self.cluster_list)):
                    sim[i] = cos_sim_l(self.wv_tfidf[idx, :], self.cluster_list[i].center_1) * self.gamma + \
                        cos_sim_l(self.lda_doc_topic[idx, :], self.cluster_list[i].center_2) * (1 - self.gamma)
                max_sim = np.max(sim)
                max_sim_id = np.argmax(sim)
                if max_sim > self.thresh: #相似度高于阈值, 纳入原有类
                    self.cluster_list[max_sim_id].add_node(
                        node_id=idx,
                        wv_vec=self.wv_tfidf[idx, :], 
                        lda_vec=self.lda_doc_topic[idx, :]
                    )
                else: #新建一个簇
                    new_node = ClusterUnit2F()
                    new_node.add_node(
                        node_id=idx,
                        wv_vec=self.wv_tfidf[idx, :], 
                        lda_vec=self.lda_doc_topic[idx, :]
                    )
                    self.cluster_list.append(new_node)
                    del new_node
        else:
            if self.cluster_list == []:
                init_node = ClusterUnit()
                init_node.add_node(node_id=0, node_vec=self.vector_mat[0, :])
                self.cluster_list.append(init_node)
            for idx in range(1, self.doc_num):
                sim = np.zeros(len(self.cluster_list))
                for i in range(len(self.cluster_list)):
                    sim[i] = cos_sim_l(self.vector_mat[idx, :], self.cluster_list[i].center)
                max_sim = np.max(sim)
                max_sim_id = np.argmax(sim)
                if max_sim > self.thresh: #相似度高于阈值, 纳入原有类
                    self.cluster_list[max_sim_id].add_node(
                        node_id=idx,
                        node_vec=self.vector_mat[idx, :]
                    )
                else: #新建一个簇
                    new_node = ClusterUnit()
                    new_node.add_node(
                        node_id=idx,
                        node_vec=self.vector_mat[idx, :]
                    )
                    self.cluster_list.append(new_node)
                    del new_node
    """

    def clustering(self):
        if self.cluster_list == []: # 原有簇空, 输入的第0篇doc生成一个初始簇
            init_node = ClusterUnit(self.feature)
            init_node.add_node(
                node_id=self.ids[0], 
                feature_vecs=[feature_matrix[0, :] for feature_matrix in self.feature_matrixs],
                entities=self.entities[0]
            )
            self.cluster_list.append(init_node)
            start = 1
        else: # 原有簇不空, 直接把所有doc往原有簇中添加或者新建簇
            start = 0

        for idx in range(start, self.doc_num):
            sim = np.zeros(len(self.cluster_list))
            for i in range(len(self.cluster_list)): #与现有的簇比较相似度
                sims = [cos_sim_l(self.cluster_list[i].centers[j], self.feature_matrixs[j][idx, :]) for j in range(self.feature)]
                sim[i] = np.vdot(np.array(sims), self.weight)
            max_sim = np.max(sim)
            max_sim_id = np.argmax(sim)

            if max_sim > self.clust_thresh: #相似度高于阈值, 纳入原有类
                self.cluster_list[max_sim_id].add_node(
                    node_id=self.ids[idx],
                    feature_vecs=[feature_matrix[idx, :] for feature_matrix in self.feature_matrixs],
                    entities=self.entities[max_sim_id]
                )
                
            else: #新建一个簇
                new_clust = ClusterUnit(self.feature)
                new_clust.add_node(
                    node_id=self.ids[idx],
                    feature_vecs=[feature_matrix[idx, :] for feature_matrix in self.feature_matrixs],
                    entities=self.entities[max_sim_id]
                )
                self.cluster_list.append(new_clust)
                del new_clust
   
    def print_result(self):
        for idx, clusterunit in enumerate(self.cluster_list):
            print("cluster : %d"%idx)
            print(clusterunit.node_list)
        print('文章数量: %d'%self.doc_num)
        print('聚类数目: %d'%len(self.cluster_list))
        print('聚类耗时: %.5f s\n'%(self.spend_time))
    

class TimedSinglePassCluster:
    """
    加入时间窗口的Single-Pass聚类算法
    
    inputs:
    --------
    clust_thresh: float, 聚合相似度阈值
    weight: Union[float, List[float]], text_representation中各种表示方法的权重
    cluster_list: List[ClusterUnit], 预定义的聚合簇
    text_representation: Union[TextRepresentation, BigBirdTextRepresentation, BERTTextRepresentation], 文本表示的实例
    time_slices: Union[List[np.ndarray], List[list]] or 2-D np.ndarray, 标明文档的属于的时间段索引切片, 内部也已经按照时间顺序排列完毕, 如[[1, 2], [3, 4]]
    
    弃用:
    --------
        gamma: float, LDA向量和W2V&tfidf向量的融合权重
        vector_mat: np.ndarray, 输入的只有单个特征矩阵, 不可与wv_tfidf、lda_doc_topic共存
        wv_tfidf: np.ndarray, [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量
        lda_doc_topic: np.ndarray, [n_doc, lda_n_components]的np.ndarray, LDA训练的文档-主题矩阵
    
    """    
    def __init__(
        self, 
        clust_thresh: float = 0.9, 
        weight: Union[float, List[float]]=0.75, 
        text_representation: Union[TextRepresentation, BigBirdTextRepresentation, BERTTextRepresentation] = None,
        cluster_list: List[ClusterUnit] = None,
        time_slices: Union[List[np.ndarray], List[list]] = None
    ):
        self.clust_thresh = clust_thresh
        if Isinstance(weight, float):
            self.weight = np.array([weight, 1 - weight, 0]) # 1, 2项权重
        elif Isinstance(weight, List[float]) or Isinstance(weight, np.ndarray):
            self.weight = np.array(weight)
        else:
            raise TypeError("weight 参数类型错误")

        assert time_slices is not None
        self.time_slices = time_slices

        if cluster_list is not None:
            self.cluster_list = cluster_list
        else:
            self.cluster_list = []  # 聚类后簇的列表

        """
        if (vector_mat is not None) and (wv_tfidf is None) and (lda_doc_topic is None):
            self.vector_mat = vector_mat
            self.doc_num = self.vector_mat.shape[0]

            self.vector_mat_list = [self.vector_mat[time_slice] for time_slice in time_slices]

            t1 = time.time()
            self.vec_clustering()
            t2 = time.time()
        elif(vector_mat is None) and (wv_tfidf is not None) and (lda_doc_topic is not None): 
            self.wv_tfidf = wv_tfidf
            self.lda_doc_topic = lda_doc_topic
            assert self.wv_tfidf.shape[0] == self.lda_doc_topic.shape[0]
            self.doc_num = self.wv_tfidf.shape[0]

            self.wv_tfidf_list = [self.wv_tfidf[time_slice] for time_slice in time_slices]
            self.lda_doc_topic_list = [self.lda_doc_topic[time_slice] for time_slice in time_slices]

            t1 = time.time()
            self.wv_lda_clustering()
            t2 = time.time()
        elif(vector_mat is None) and (wv_tfidf is None) and (lda_doc_topic is None): 
            pass
        else:
            raise RuntimeError("vector_mat不可与wv_tfidf、lda_doc_topic共存")
        """
        if Isinstance(text_representation, TextRepresentation):
            assert text_representation.wv_tfidf.shape[0] == text_representation.lda_doc_topic.shape[0]
            self.doc_num = text_representation.wv_tfidf.shape[0]
            self.feature_matrixs = [text_representation.wv_tfidf, text_representation.lda_doc_topic] # 重要性降序
            self.feature = 2 # 特征矩阵数目
            
        elif Isinstance(text_representation, BigBirdTextRepresentation):
            assert text_representation.big_bird_rp.shape[0] == text_representation.big_bird_rp.shape[0]
            self.doc_num = text_representation.wv_tfidf.shape[0]
            self.feature_matrixs = [text_representation.big_bird_rp, text_representation.wv_tfidf] # 重要性降序
            self.feature = 2 # 特征矩阵数目
        elif Isinstance(text_representation, BERTTextRepresentation):
            assert text_representation.bert_rp.shape[0] == text_representation.bert_rp.shape[0]
            self.doc_num = text_representation.wv_tfidf.shape[0]
            self.feature_matrixs = [text_representation.bert_rp, text_representation.wv_tfidf] # 重要性降序
            self.feature = 2 # 特征矩阵数目
        else:
            raise TypeError("text_representation 类型错误")
        self.entities = text_representation.entities
        self.ids = text_representation.ids
        assert len(self.ids) == self.doc_num

        t1 = time.time()
        self.clustering()
        t2 = time.time()

        self.spend_time = t2 - t1  # 聚类花费的时间
        self.cluster_num = len(self.cluster_list) # 聚类完成后 簇的个数
    
    """
    def wv_lda_clustering(self):
        if self.cluster_list == []:
            init_node = ClusterUnitWVLDA()
            init_idx = self.time_slices[0][0]
            init_node.add_node(node_id=init_idx, wv_vec=self.wv_tfidf[init_idx, :], lda_vec=self.lda_doc_topic[init_idx, :])
            self.cluster_list.append(init_node)
        for t in range(len(self.time_slices)):
            for idx in self.time_slices[t]:
                if idx == init_idx:
                    continue
                sim = np.zeros(len(self.cluster_list))
                for i in range(len(self.cluster_list)):
                    sim[i] = cos_sim_l(self.wv_tfidf[idx, :], self.cluster_list[i].center_wv) * self.gamma + \
                        cos_sim_l(self.lda_doc_topic[idx, :], self.cluster_list[i].center_lda) * (1 - self.gamma)
                max_sim, max_sim_id = np.max(sim), np.argmax(sim)
                    
                if max_sim > self.thresh: #相似度高于阈值, 纳入原有类
                    self.cluster_list[max_sim_id].add_node(
                        node_id=idx,
                        wv_vec=self.wv_tfidf[idx, :], 
                        lda_vec=self.lda_doc_topic[idx, :]
                    )
                else: #新建一个簇
                    new_node = ClusterUnitWVLDA()
                    new_node.add_node(
                        node_id=idx,
                        wv_vec=self.wv_tfidf[idx, :], 
                        lda_vec=self.lda_doc_topic[idx, :]
                    )
                    self.cluster_list.append(new_node)
                    del new_node

    def vec_clustering(self):
        if self.cluster_list == []:
            init_node = ClusterUnit()
            init_idx = self.time_slices[0][0]
            init_node.add_node(node_id=init_idx, node_vec=self.vector_mat[init_idx, :])
            self.cluster_list.append(init_node)

        for t in range(len(self.time_slices)):
            for idx in self.time_slices[t]:
                if idx == init_idx:
                    continue
                sim = np.zeros(len(self.cluster_list))
                for i in range(len(self.cluster_list)):
                    sim[i] = cos_sim_l(self.vector_mat[idx, :], self.cluster_list[i].center)
                    max_sim, max_sim_id = np.max(sim), np.argmax(sim)

                if max_sim > self.thresh: #相似度高于阈值, 纳入原有类
                    self.cluster_list[max_sim_id].add_node(
                        node_id=idx,
                        wv_vec=self.wv_tfidf[idx, :], 
                        lda_vec=self.lda_doc_topic[idx, :]
                    )
                else: #新建一个簇
                    new_node = ClusterUnitWVLDA()
                    new_node.add_node(
                        node_id=idx,
                        wv_vec=self.wv_tfidf[idx, :], 
                        lda_vec=self.lda_doc_topic[idx, :]
                    )
                    self.cluster_list.append(new_node)
                    del new_node
    """

    def clustering(self):
        if self.cluster_list == []: #无预定义簇, 建立初始簇
            init_node = ClusterUnit(self.feature)
            init_idx = self.time_slices[0][0] #输入顺序中第0个时间切片的第0篇文章
            init_node.add_node(
                node_id=self.ids[init_idx], 
                feature_vecs=[feature_matrix[init_idx, :] for feature_matrix in self.feature_matrixs],
                entities = self.entities[init_idx]
            )
            self.cluster_list.append(init_node)
            init_flag = True

        else: #存在预定义定义簇, 直接遍历
            init_flag = False
            init_idx = -1

        
        for t in range(len(self.time_slices)):
            for idx in self.time_slices[t]:
                if init_flag and idx == init_idx: #跳过初始化簇的结点
                    continue

                sim = np.zeros(len(self.cluster_list))
                for i in range(len(self.cluster_list)):
                    sims = [cos_sim_l(self.cluster_list[i].centers[j], self.feature_matrixs[j][idx, :]) for j in range(self.feature)]
                    sim[i] = np.vdot(np.array(sims), self.weight)

                max_sim = np.max(sim)
                max_sim_id = np.argmax(sim)   

                if max_sim > self.clust_thresh: #相似度高于阈值, 纳入原有类
                    self.cluster_list[max_sim_id].add_node(
                        node_id=self.ids[idx],
                        feature_vecs=[feature_matrix[idx, :] for feature_matrix in self.feature_matrixs],
                        entities=self.entities[max_sim_id]
                    )
                
                else:#新建一个簇
                    new_clust = ClusterUnit(self.feature)
                    new_clust.add_node(
                        node_id=self.ids[idx],
                        feature_vecs=[feature_matrix[idx, :] for feature_matrix in self.feature_matrixs],
                        entities=self.entities[max_sim_id]
                    )
                    self.cluster_list.append(new_clust)
                    del new_clust
        # self.cluster_list = [[j for j in len(self.cluster_list) if i in self.cluster_list[j].node_list] for i in range(self.doc_num)]
        pass


    def print_result(self):
        for idx, clusterunit in enumerate(self.cluster_list):
            print("cluster : %d"%idx)
            print(clusterunit.node_list)
        print('文章数量: %d'%self.doc_num)
        print('聚类数目: %d'%len(self.cluster_list))
        print('聚类耗时: %.5f s\n'%(self.spend_time))

