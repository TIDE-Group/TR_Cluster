import numpy as np
import scipy.sparse as sp
import warnings
import time
from gensim import corpora
warnings.filterwarnings('ignore')

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

class ClusterUnitWVLDA:
    def __init__(self):
        self.node_list = []
        self.center_wv = None
        self.center_lda = None
    
    def __len__(self):
        return len(self.node_list)
    
    def add_node(self, node_id:int, wv_vec:np.ndarray, lda_vec:np.ndarray):
        self.node_list.append(node_id)
        try:
            # self.center = ((len(self.node_list) - 1) * self.center + node_vec) / len(self.node_list)
            self.center_wv = ((len(self.node_list) - 1) * self.center_wv + wv_vec) / len(self.node_list)
            self.center_lda = ((len(self.node_list) - 1) * self.center_lda + lda_vec) / len(self.node_list)
        except TypeError:
            # self.center = node_vec # 初始化质心
            self.center_wv = wv_vec
            self.center_lda = lda_vec
    
    def remove_node(self, node_id:int, wv_vec:np.ndarray, lda_vec:np.ndarray):
        try:
            self.center_wv = (len(self.node_list) * self.center_wv - wv_vec) / (len(self.node_list) - 1)
            self.center_lda = (len(self.node_list) * self.center_lda - lda_vec) / (len(self.node_list) - 1)
            self.node_list.remove(node_id)
        except ValueError:
            raise ValueError("%d 不在这个聚类中"%node_id)
    
    def move_node(self, 
        node_id:int, wv_vec: np.ndarray, lda_vec: np.ndarray, 
        moved_cluster
    ): #将本簇的结点移到另一个簇
        self.remove_node(node_id)
        moved_cluster.add_node(node_id, wv_vec, lda_vec)
    
    def add_cluster(
        self,
        added_cluster
    ): #将本簇完全添加到另一个簇
        try:
            added_cluster.center_wv = (len(added_cluster.node_list) * added_cluster.center_wv + len(self.node_list) * self.center_wv) / \
                (len(added_cluster.node_list) + len(self.node_list))
            added_cluster.center_lda = (len(added_cluster.node_list) * added_cluster.center_lda + len(self.node_list) * self.center_lda) / \
                (len(added_cluster.node_list) + len(self.node_list))
        except TypeError:
            added_cluster.center_wv = self.center_wv
            added_cluster.center_lda = self.center_lda
        # for node_id in self.node_list:
        #     added_cluster.node_list.append(node_id)
        added_cluster.node_list.extend(self.node_list)
    
    @staticmethod
    def union_cluster(cluster_1, cluster_2): #静态方法，合并两个簇
        assert len(cluster_1) > 0
        assert len(cluster_2) > 0
        union = ClusterUnitWVLDA()
        union.node_list = cluster_1.node_list
        union.node_list.extend(cluster_2.node_list)
        union.center_wv = (len(cluster_1) * cluster_1.center_wv + len(cluster_2) * cluster_2.center_wv) / (len(cluster_1) + len(cluster_2))
        union.center_lda = (len(cluster_1) * cluster_1.center_lda + len(cluster_2) * cluster_2.center_lda) / (len(cluster_1) + len(cluster_2))
        
        return union

def cos_sim(v_a : np.ndarray, v_b : np.ndarray):
    cos = np.vdot(v_a, v_b) / (np.linalg.norm(v_a, 2) * np.linalg(v_b, 2))
    return cos

cos_sim_l = lambda v_a, v_b: np.vdot(v_a, v_b) / (np.linalg.norm(v_a, 2) * np.linalg.norm(v_b, 2))

class SinglePassCluster:
    """
    Single-Pass聚类算法
    inputs:
    --------
    thresh: float, 相似度阈值
    gamma: float, LDA向量和W2V&tfidf向量的融合权重
    cluster_list: list[ClusterUnit] or list[ClusterUnitWVLDA], 预定义的聚合簇
    vector_mat: np.ndarray, 输入的只有单个特征矩阵, 不可与wv_tfidf、lda_doc_topic共存
    wv_tfidf: np.ndarray, [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量
    lda_doc_topic: np.ndarray, [n_doc, lda_n_components]的np.ndarray, LDA训练的文档-主题矩阵
    """
    def __init__(
        self, 
        thresh:float = 0.9, 
        gamma: float = 0.5,
        cluster_list: list = None,
        vector_mat: np.ndarray = None,
        wv_tfidf: np.ndarray = None,
        lda_doc_topic: np.ndarray = None
    ):
        self.thresh = thresh
        self.gamma = gamma
        if cluster_list is not None:
            self.cluster_list = cluster_list
        else:
            self.cluster_list = []  # 聚类后簇的列表
        
        if (vector_mat is not None) and (wv_tfidf is None) and (lda_doc_topic is None):
            self.vector_mat = vector_mat
            self.doc_num = self.vector_mat.shape[0]
            t1 = time.time()
            self.clustering(WVLDA=False)
            t2 = time.time()
        elif(vector_mat is None) and (wv_tfidf is not None) and (lda_doc_topic is not None): 
            self.wv_tfidf = wv_tfidf
            self.lda_doc_topic = lda_doc_topic
            assert self.wv_tfidf.shape[0] == self.lda_doc_topic.shape[0]
            self.doc_num = self.wv_tfidf.shape[0]
            t1 = time.time()
            self.clustering(WVLDA=True)
            t2 = time.time()
        elif(vector_mat is None) and (wv_tfidf is None) and (lda_doc_topic is None): 
            pass
        else:
            raise RuntimeError("vector_mat不可与wv_tfidf、lda_doc_topic共存")
        
        self.spend_time = t2 - t1  # 聚类花费的时间
        self.cluster_num = len(self.cluster_list) # 聚类完成后 簇的个数
    
    def clustering(self, WVLDA=True):
        if WVLDA:
            if self.cluster_list == []:
                init_node = ClusterUnitWVLDA()
                init_node.add_node(node_id=0, wv_vec=self.wv_tfidf[0, :], lda_vec=self.lda_doc_topic[0, :])
                self.cluster_list.append(init_node)
            for idx in range(1, self.doc_num):
                sim = np.zeros(len(self.cluster_list))
                for i in range(len(self.cluster_list)):
                    sim[i] = cos_sim_l(self.wv_tfidf[idx, :], self.cluster_list[i].center_wv) * self.gamma + \
                        cos_sim_l(self.lda_doc_topic[idx, :], self.cluster_list[i].center_lda) * (1 - self.gamma)
                max_sim = np.max(sim)
                max_sim_id = np.argmax(sim)
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

    def print_result(self):
        for idx, clusterunit in enumerate(self.cluster_list):
            print("cluster : %d"%idx)
            print(clusterunit.node_list)
        print('文章数量: %d'%self.doc_num)
        print('聚类数目: %d'%len(self.cluster_list))
        print('聚类耗时: %.5f s\n'%(self.spend_time / 1000))
    

class TimedSinglePassCluster:
    """
    加入时间窗口的Single-Pass聚类算法
    inputs:
    --------
    thresh: float, 聚合相似度阈值
    gamma: float, LDA向量和W2V&tfidf向量的融合权重
    cluster_list: list[ClusterUnit] or list[ClusterUnitWVLDA], 预定义的聚合簇
    vector_mat: np.ndarray, 输入的只有单个特征矩阵, 不可与wv_tfidf、lda_doc_topic共存
    wv_tfidf: np.ndarray, [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量
    lda_doc_topic: np.ndarray, [n_doc, lda_n_components]的np.ndarray, LDA训练的文档-主题矩阵
    time_slices: list[np.ndarray] or list[list] or 2-D np.ndarray, 标明文档的时间索引切片, 内部也已经按照时间顺序排列完毕, 如[[1, 2], [3, 4]]
    
    """    
    def __init__(
        self, 
        thresh:float = 0.9, 
        gamma: float = 0.5,
        cluster_list: list = None,
        vector_mat: np.ndarray = None,
        wv_tfidf: np.ndarray = None,
        lda_doc_topic: np.ndarray = None,
        time_slices: list = None
    ):
        self.thresh = thresh
        self.gamma = gamma
        assert time_slices is not None
        self.time_slices = time_slices

        if cluster_list is not None:
            self.cluster_list = cluster_list
        else:
            self.cluster_list = []  # 聚类后簇的列表

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
        
        self.spend_time = t2 - t1  # 聚类花费的时间
        self.cluster_num = len(self.cluster_list) # 聚类完成后 簇的个数
    
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
    
    def print_result(self):
        for idx, clusterunit in enumerate(self.cluster_list):
            print("cluster : %d"%idx)
            print(clusterunit.node_list)
        print('文章数量: %d'%self.doc_num)
        print('聚类数目: %d'%len(self.cluster_list))
        print('聚类耗时: %.5f s\n'%(self.spend_time / 1000))

