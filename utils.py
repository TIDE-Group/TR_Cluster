from typing import List, Union, Set
from collections import Counter
from textrepresentation import TextRepresentation, BigBirdTextRepresentation, BERTTextRepresentation
import numpy as np
import scipy.sparse as sp
import pandas as pd
import warnings
import time
from gensim import corpora
from sklearn.metrics import accuracy_score, precision_score, recall_score,  f1_score  # 计算f1
import typesentry
from torch.utils.data.dataset import Dataset
import torch
from torch import cuda
tc1 = typesentry.Config()
Isinstance = tc1.is_type
warnings.filterwarnings('ignore')


class ClusterUnit(object):
    """
    簇类
    feature: int, 特征矩阵数

    attribute:
    --------
    centers: List[np.ndarray(feature_dim) * feature] 
    entities: set(tuple), 命名实体集合
    """ 
    def __init__(self, feature:int=3, entities:set=None):
        self.feature = feature
        self.node_list = []
        self.centers = [None for i in range(feature)]
        self.entities = entities if entities else set()
    
    def __len__(self):
        return len(self.node_list)

    def add_node(self, node_id:int, feature_vecs:List[np.ndarray], entities:set):
        assert self.feature == len(feature_vecs)
        self.node_list.append(node_id)
        self.entities = self.entities.union(entities)
        try:
            for i in range(self.feature):
                self.centers[i] = ((len(self.node_list) - 1) * self.centers[i] + feature_vecs[i]) / len(self.node_list)
        except TypeError:
            self.centers = feature_vecs

    def remove_node(self, node_id:int, feature_vecs:List[np.ndarray], entities:set):
        assert self.feature == len(feature_vecs)
        self.entities = self.entities - entities
        try:
            for i in range(self.feature):
                self.centers[i] = (len(self.node_list) * self.centers[i] - feature_vecs[i]) / (len(self.node_list) - 1)
            self.node_list.remove(node_id)
        except ValueError:
            raise ValueError("%d 不在这个簇中"%node_id)

    def move_node(self, 
        node_id:int, feature_vecs:List[np.ndarray], entities:set, 
        moved_cluster
    ): #将本簇的结点移到另一个簇
        try:
            self.remove_node(node_id, entities)
            moved_cluster.add_node(node_id, feature_vecs, entities)
        except ValueError:
            raise ValueError("%d 不在这个簇中"%node_id)

    def add_cluster(
        self,
        added_cluster
    ): #将本簇完全添加到另一个簇
        try:
            for i in range(added_cluster.feature):
                added_cluster.centers[i] = (len(added_cluster.node_list) * added_cluster.centers[i] + len(self.node_list) * self.centers[i]) / \
                    (len(added_cluster.node_list) + len(self.node_list))
        except TypeError:
            for i in range(added_cluster.feature):
                added_cluster.centers[i] = self.centers[i]
        # for node_id in self.node_list:
        #     added_cluster.node_list.append(node_id)
        added_cluster.node_list.extend(self.node_list)
        added_cluster.entities = added_cluster.entities.union(self.entities)

    @staticmethod
    def union_cluster(cluster_1, cluster_2): #静态方法，合并两个簇
        assert len(cluster_1) > 0
        assert len(cluster_2) > 0
        assert cluster_1.feature == cluster_2.feature
        union = ClusterUnit(cluster_1.feature)
        union.node_list = cluster_1.node_list
        union.node_list.extend(cluster_2.node_list)
        union.entities = cluster_1.entities.union(cluster_2.entities)
        for i in range(union.feature):
            union.centers[i] = (len(cluster_1) * cluster_1.centers[i] + len(cluster_2) * cluster_2.centers[i]) / (len(cluster_1) + len(cluster_2))

        return union


class ClassifierDataSet(Dataset):
    def __init__(self, texts:List[str], labels:Union[List[int], np.ndarray]):
        super(ClassifierDataSet, self).__init__()
        assert len(texts) == len(labels), "文本和标签数量不对应"
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return self.texts[index], torch.LongTensor([self.labels[index]])
    
def classifier_collate_fn(data):
    texts, labels = map(list, zip(*data))
    labels = torch.cat(labels, dim=0)
    if cuda.is_available():
        labels = labels.cuda()

    return texts, labels

def calMacro(predict_results, true_results):
    acc = accuracy_score(true_results.astype('int'), predict_results.astype('int'))
    Pmacro = precision_score(true_results.astype('int'), predict_results.astype('int'), average='macro')
    Rmacro = recall_score(true_results.astype('int'), predict_results.astype('int'), average='macro')
    f1 = f1_score(true_results.astype('int'), predict_results.astype('int'), average='macro')
    # Calculate f1 score
    return (acc, Pmacro, Rmacro, f1)

def calmetrics(predict_results, true_results):
    acc = accuracy_score(true_results.astype('int'), predict_results.astype('int'))
    Pmacro = precision_score(true_results.astype('int'), predict_results.astype('int'), average=None)
    Rmacro = recall_score(true_results.astype('int'), predict_results.astype('int'), average=None)
    f1 = f1_score(true_results.astype('int'), predict_results.astype('int'), average=None)
    # Calculate f1 score
    return (acc, Pmacro, Rmacro, f1)

def eval_clustering_results(labels:Union[List[int], np.ndarray], clusting_result:List[List[int]], ave=True):
    labels = np.array(labels, dtype=int)
    clusting_labels = [labels[cluster] for cluster in clusting_result]
    clusting_label_counter = [Counter(clusting_label) for clusting_label in clusting_labels]
    most_common_label = [clusting_label.most_common(1)[0][0] for clusting_label in clusting_label_counter]
    pred_labels = np.zeros_like(labels, dtype=int)
    assert len(clusting_result) == len(most_common_label)
    for clust, common_label in zip(clusting_result, most_common_label):
        pred_labels[clust] = common_label
    if ave:
        acc, Pmacro, Rmacro, f1 = calMacro(pred_labels, labels)
        return acc, Pmacro, Rmacro, f1, pred_labels, labels
    else:
        acc, Pmacro, Rmacro, f1 = calmetrics(pred_labels, labels)
        return acc, Pmacro, Rmacro, f1, pred_labels, labels

def set_iou(set_a:set, set_b:set):
    if len(set_a) == 0 or len(set_b) == 0:
        return 0.0
    else:
        return len(set_a.intersection(set_b)) / len(set_a.union(set_b))

def convertid2index(df:pd.DataFrame, cluster_unit_ids:Union[np.ndarray, List[int]]):
    
    return np.concatenate([df[df['id'] == id].index.values for id in cluster_unit_ids])
    


     
