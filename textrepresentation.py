from typing import Text, Union, List
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from gensim.models import Word2Vec
import numpy as np
import scipy.sparse as sp

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')
from transformers import BigBirdTokenizer, BigBirdModel, BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.cuda as cuda
import random
import math
import typesentry
import os
import stanza
import re

tc1 = typesentry.Config()
Isinstance = tc1.is_type

from tqdm import tqdm

class NERVocab(object):
    def __init__(self, id2entity:dict, entity2id:dict):
        self.id2entity = id2entity
        self.entity2id = entity2id
    
    def __getitem__(self, value:Union[int, tuple, str]):
        if Isinstance(value, int):
            return self.id2entity[value]
        elif Isinstance(value, tuple):
            return self.entity2id[str(value)]
        elif Isinstance(value, str):
            return self.entity2id[value]
        else:
            raise TypeError("type value error, value must belog to Union[int, tuple, str]")


class TextRepresentation(object):
    """
    文本表示的类
    input:
    --------
    ```
    texts: list[str], 每一个str为一篇文本
    stopwords: list, 包含所有的停用词
    labels: List[int], 每篇文章对应的label, 求结果时每一类的lab可能不同
    wv_dim: int, 词向量的维度
    lda_n_compontents: int, LDA模型的主题数目
    wv_iter: Word2Vec模型迭代次数, 默认为5
    lda_max_iter: LDA模型迭代次数, 默认为10
    tf_idf_max_df, tfidf_min_df: float in range[0, 1], 参见tf-idf说明
    wv_min_count: int, W2V中纳入计算的最小词频, 为保证W2V和tf-idf模型匹配, 默认为1, 不建议修改
    jobs: int, 训练线程数
    random_seed: int, 随机种子
    stanza_path: str, stanza模型的路径, 默认为'../default/'
    sparse: bool, wv_tfidf是否选择为稀疏矩阵, 默认为False
    ```
  
    output:
    --------
    ```
    wv_tfidf: [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量

    lda_doc_topic: [n_doc, lda_n_components]的np.ndarray, LDA训练的文档-主题矩阵
        
    lda_topic_term: [lda_n_components, vocab_size]的np.ndarray, LDA训练得到的主题-词矩阵

    dictionary: class: gensim.corpora.Dictionary, 经过预处理后的文本词典

    entities: List[Set], 每篇文章的命名实体集合
    ```
    """
    def __init__(
        self,
        texts: list = None,
        stopwords: list = None,
        labels: List[int] = None,
        wv_dim: int = 100,
        lda_n_components: int = 100,
        wv_iter: int = 10,
        lda_max_iter: int = 10,
        tfidf_max_df: float = 1.0,
        tfidf_min_df = 1,
        wv_min_count: int = 1,
        jobs: int = 12,
        random_seed: int = 1,
        stanza_path: str = '../default/', 
        sparse: bool = False,
        ner=True
    ):
        """
        input:
        --------
        ```
        texts: list[str], 每一个str为一篇文本
        stopwords: list, 包含所有的停用词
        labels: List[int], 每篇文章对应的label, 求结果时每一类的lab可能不同
        wv_dim: int, 词向量的维度
        lda_n_compontents: int, LDA模型的主题数目
        wv_iter: Word2Vec模型迭代次数, 默认为5
        lda_max_iter: LDA模型迭代次数, 默认为10
        tf_idf_max_df, tfidf_min_df: float in range[0, 1], 参见tf-idf说明
        wv_min_count: int, W2V中纳入计算的最小词频, 为保证W2V和tf-idf模型匹配, 默认为1, 不建议修改
        jobs: int, 训练线程数
        random_seed: int, 随机种子
        stanza_path: str, stanza模型的路径, 默认为'../default/'
        sparse: bool, wv_tfidf是否选择为稀疏矩阵, 默认为False
        ```
  
        output:
        --------
        ```
        wv_tfidf: [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量

        lda_doc_topic: [n_doc, lda_n_components]的np.ndarray, LDA训练的文档-主题矩阵
        
        lda_topic_term: [lda_n_components, vocab_size]的np.ndarray, LDA训练得到的主题-词矩阵

        dictionary: class: gensim.corpora.Dictionary, 经过预处理后的文本词典

        entities: [n_doc, n_entities]的np.ndarray, 所有文章的实体集合的tf-idf特征
        ```
        """
        super(TextRepresentation, self).__init__()
        # assert len(texts) == len(labels), "label与文章数量不相等"
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        cuda.manual_seed(random_seed)
        os.environ['PYTHONHASHSEED '] = str(random_seed)
        
        self.wv_dim = wv_dim
        self.lda_n_component = lda_n_components
        self.sparse = sparse
        self._labels = labels
        if stopwords is None:
            stopwords = get_stop_words('en')
        
        if texts is not None:
            texts = self._preproceeding(texts, stopwords)
            self._texts = texts
            self.dictionary = corpora.Dictionary(texts)
            self.tfidf_tsf = TfidfVectorizer(
                stop_words=stopwords,
                min_df=tfidf_min_df,
                max_df=tfidf_max_df,
                vocabulary=self.dictionary.token2id
            )
            self.w2vmodel = Word2Vec(
                sentences = texts,
                size=wv_dim,
                min_count=wv_min_count,
                iter=wv_iter,
                sg=1, hs=1,
                workers=jobs,
                seed=random_seed
            )
        
        else:
            self._texts = None
            self.dictionary = None
            self.tfidf_tsf = TfidfVectorizer(
                stop_words=stopwords,
                min_df=tfidf_min_df,
                max_df=tfidf_max_df
            )
        
            self.w2vmodel = Word2Vec(
                size=wv_dim,
                min_count=wv_min_count,
                iter=wv_iter,
                sg=1, hs=1,
                workers=jobs,
                seed=random_seed
            )
        self.lda_model = LatentDirichletAllocation(
            n_components=lda_n_components,
            learning_method='batch', 
            n_jobs=jobs,
            max_iter=lda_max_iter,
            random_state=random_seed
        )
        self.ner_processer = stanza.Pipeline(
            lang='en', dir=stanza_path, 
            processors='tokenize, ner', 
            use_gpu=True
        )
        
        if texts is not None:
            texts = [' '.join(text) for text in texts]

            if ner:
                self.entities = []
                with torch.no_grad():
                    for text in tqdm(texts, total=len(texts), desc='calculating NER'):
                        self.entities.append(set(self._ner_result_extractor(self.ner_processer(text))))
            else:
                # self.entities = np.random.randn(len(texts), 256)
                self.entities = [set() for _ in range(len(texts))]
            tf_idf = self.tfidf_tsf.fit_transform(texts) # [doc_num, word_cnt]
            wv = self.w2vmodel.wv
            idx = [self.dictionary[i] for i in range(len(self.dictionary))]
            wv = wv[idx] # [word_cnt, wv_dim]
            if not sparse:
                # self.wv_tfidf = np.matmul(tf_idf.toarray(), wv)
                wv = sp.csr_matrix(wv)
                self.wv_tfidf = tf_idf * wv # [doc_num, wv_dim]
                self.wv_tfidf = self.wv_tfidf.toarray()
            else:
                wv = sp.csr_matrix(wv)
                self.wv_tfidf = tf_idf * wv
            self.lda_doc_topic = self.lda_model.fit_transform(tf_idf)
            self.lda_topic_term = self.lda_model.components_
        
    def _ner_result_extractor(self, result):
        return [(ent.text, ent.type) for sent in result.sentences for ent in sent.ents]

    def build_entity_tfidf(self, doc_entities:List[list]):
        doc_entities_str = [[re.sub(r'\\', '',str(entity)) for entity in entities] for entities in doc_entities]
        vocab = corpora.Dictionary(doc_entities_str)

        doc_entities_encode = [[str(vocab.token2id[entity]) for entity in entities] for entities in doc_entities_str]
        doc_entities_encode = [' '.join(entities) for entities in doc_entities_encode]

        token2id = {str(i):i for i in range(len(vocab))}

        tfidf_tsf = TfidfVectorizer(
            vocabulary=token2id,
            min_df=0.0,
            max_df=1.0
        )

        entities_tfidf_mat = tfidf_tsf.fit_transform(doc_entities_encode).toarray()
        
        return entities_tfidf_mat, vocab, tfidf_tsf

    def _preproceeding(self, texts:list, stopwords:list):
        """
        预处理输入的list[str] text

        input:
        --------
        texts: list[str]
        stopwords: list[str]

        output:
        --------
        list[str], 得到预处理的文本
        """
        tokenizer = RegexpTokenizer(r'\w+')
        p_stemmer = PorterStemmer()
        def __preproceeding(text:str):
            text = text.lower()
            tokens = tokenizer.tokenize(text)
            tokens = list(filter(lambda token: token not in stopwords, tokens))
            tokens = [p_stemmer.stem(token) for token in tokens]
            return tokens
        
        texts = [__preproceeding(text) for text in texts]
        return texts
    
    def update(self, texts:list, labels:List[int], stopwords:list=None, ner=True):
        """
        对输入的文本进行训练
        input:
        --------
        texts: list[str]
        labels: List[int]
        stopwords: list[str]

        output:
        --------
        None(self)
        """
        assert len(texts) == len(labels), "label与文章数量不相等"
        if stopwords is None:
            stopwords = get_stop_words('en')

        texts = self._preproceeding(texts, stopwords)
        if self.dictionary is None:
            self._texts = texts
            self._labels = labels
            self.dictionary = corpora.Dictionary(texts)
            self.w2vmodel.build_vocab(texts)
        else:
            self._texts.extend(texts)
            self._labels.extend(labels)
            # print(len(self.dictionary))
            self.dictionary.add_documents(texts)
            # print(len(self.dictionary))
            self.w2vmodel.clear_sims()
            self.w2vmodel.build_vocab(self._texts, update=True)
        self.tfidf_tsf.vocabulary = self.dictionary.token2id
        Texts = [' '.join(text) for text in self._texts]
        if ner:
            if not hasattr(self, 'entities'):
                self.entities = []
            with torch.no_grad():
                for text in tqdm(texts, total=len(texts), desc='calculating NER'):
                    self.entities.append(set(self._ner_result_extractor(self.ner_processer(text))))
        else:
            # self.entities = [set() for _ in range(len(Texts))]
            if not hasattr(self, 'entities'):
                self.entities = [set() for _ in range(len(Texts))]
            else:
                for _ in range(len(Texts)):
                    self.entities.append(set())
        # self.entities = [self._ner_result_extractor(self.ner_processer(text)) for text in texts]
        # self.entities, self.entities_vocab, self.entities_tfidf_tsf = self.build_entity_tfidf(self.entities)
        
        tf_idf = self.tfidf_tsf.fit_transform(Texts)
        
        self.w2vmodel.train(self._texts, total_examples=self.w2vmodel.corpus_count, epochs=self.w2vmodel.iter)
        wv = self.w2vmodel.wv
        idx = [self.dictionary[i] for i in range(len(self.dictionary))]
        wv = wv[idx]
        if not self.sparse:
            # self.wv_tfidf = np.matmul(tf_idf.toarray(), wv)
            wv = sp.csr_matrix(wv)
            self.wv_tfidf = tf_idf * wv
            self.wv_tfidf = self.wv_tfidf.toarray()
        else:
            wv = sp.csr_matrix(wv)
            self.wv_tfidf = tf_idf * wv
        self.lda_doc_topic = self.lda_model.fit_transform(tf_idf)
        self.lda_topic_term = self.lda_model.components_

class BigBirdTextRepresentation(object):
    """
    采用BigBird模型的文本表示
    input:
    --------
    ```
    texts: list[str], 每一个str为一篇文本
    stopwords: list, 包含所有的停用词
    labels: List[int], 每篇文章对应的label, 求结果时每一类的lab可能不同
    wv_dim: int, 词向量的维度
    batch_size: int, BigBird模型的batchsize
    wv_iter: Word2Vec模型迭代次数, 默认为5
    tf_idf_max_df, tfidf_min_df: float in range[0, 1], 参见tf-idf说明
    wv_min_count: int, W2V中纳入计算的最小词频, 为保证W2V和tf-idf模型匹配, 默认为1, 不建议修改
    jobs: int, W2V训练线程数
    random_seed: int, 随机种子
    stanza_path: str, stanza模型的路径, 默认为'../default/'
    sparse: bool, wv_tfidf的输出是否选择为稀疏矩阵, 默认为False
    ```
  
    components:
    --------
    ```
    wv_tfidf: [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量

    big_bird_rp: [n_doc, 768]的np.ndarray, BigBird模型的pooler_output

    dictionary: class: gensim.corpora.Dictionary, 经过预处理后的文本词典

    entities: List[Set], 每篇文章的命名实体集合
    ```
    """
    def __init__(
        self,
        texts: list = None,
        stopwords: list = None,
        labels:List[int] = None,
        wv_dim: int = 100,
        batch_size: int = 10,
        wv_iter: int = 10,
        tfidf_max_df: float = 1.0,
        tfidf_min_df = 1,
        wv_min_count: int = 1,
        jobs: int = 1,
        random_seed: int = 1,
        stanza_path: str = '../default/', 
        sparse: bool = False,
        ner=True
    ):
        """
        采用BigBird模型的文本表示
        input:
        --------
        ```
        texts: list[str], 每一个str为一篇文本
        stopwords: list, 包含所有的停用词
        labels: List[int], 每篇文章对应的label, 求结果时每一类的lab可能不同
        wv_dim: int, 词向量的维度
        batch_size: int, BigBird模型的batchsize, 默认为10
        wv_iter: Word2Vec模型迭代次数, 默认为5
        tf_idf_max_df, tfidf_min_df: float in range[0, 1], 参见tf-idf说明
        wv_min_count: int, W2V中纳入计算的最小词频, 为保证W2V和tf-idf模型匹配, 默认为1, 不建议修改
        jobs: int, W2V训练线程数
        random_seed: int, 随机种子
        stanza_path: str, stanza模型的路径, 默认为'../default/'
        sparse: bool, wv_tfidf的输出是否选择为稀疏矩阵, 默认为False
        ```
    
        components:
        --------
        ```
        wv_tfidf: [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量

        big_bird_rp: [n_doc, 768]的np.ndarray, BigBird模型的pooler_output

        dictionary: class: gensim.corpora.Dictionary, 经过预处理后的文本词典

        entities: List[Set], 每篇文章的命名实体集合
        ```
        """
        
        super(BigBirdTextRepresentation, self).__init__()
        assert len(texts) == len(labels), "label与文章数量不相等"
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        cuda.manual_seed(random_seed)
        os.environ['PYTHONHASHSEED '] = str(random_seed)
        self.wv_dim = wv_dim
        self.batch_size = batch_size
        self._labels = labels
        self.sparse = sparse
        self.big_bird_tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

        self.big_bird_model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
        self.big_bird_model = self.big_bird_model.eval().cuda()


        if stopwords is None:
            stopwords = get_stop_words('en')
        
        if texts is not None:
            texts = self._preproceeding(texts, stopwords)
            self._texts = texts
            self.doc_num = len(texts)
            self.big_bird_rp = np.zeros((self.doc_num, 768))

            self.dictionary = corpora.Dictionary(texts)
            self.tfidf_tsf = TfidfVectorizer(
                stop_words=stopwords,
                min_df=tfidf_min_df,
                max_df=tfidf_max_df,
                vocabulary=self.dictionary.token2id
            )
            self.w2vmodel = Word2Vec(
                sentences = texts,
                size=wv_dim,
                min_count=wv_min_count,
                iter=wv_iter,
                sg=1, hs=1,
                workers=jobs,
                seed=random_seed
            )
        
        else:
            self._texts = None
            self.dictionary = None
            self.tfidf_tsf = TfidfVectorizer(
                stop_words=stopwords,
                min_df=tfidf_min_df,
                max_df=tfidf_max_df
            )
        
            self.w2vmodel = Word2Vec(
                size=wv_dim,
                min_count=wv_min_count,
                iter=wv_iter,
                sg=1, hs=1,
                workers=jobs,
                seed=random_seed
            )

        self.ner_processer = stanza.Pipeline(
            lang='en', dir=stanza_path, 
            processors='tokenize, ner', 
            use_gpu=True
        )
        
        if texts is not None:
            # max_lenth = np.max([len(text) for text in texts])
            # padding = [['<pad>'] * (max_lenth - len(text)) for text in texts]
            Texts = [' '.join(text) for text in texts]
            if ner:
                self.entities = []
                with torch.no_grad():
                    for text in tqdm(Texts, desc='calculating NER'):
                        self.entities.append(set(self._ner_result_extractor(self.ner_processer(text))))
                # self.entities = [self._ner_result_extractor(self.ner_processer(text)) for text in Texts]
                # self.entities, self.entities_vocab, self.entities_tfidf_tsf = self.build_entity_tfidf(self.entities)
            else:
                # self.entities = np.random.randn(len(Texts), 256)
                self.entities = [set() for _ in range(len(Texts))]

            tokens_pt = self.big_bird_tokenizer(Texts, return_tensors='pt', padding=True)

            tf_idf = self.tfidf_tsf.fit_transform(Texts) # [doc_num, word_cnt]
            wv = self.w2vmodel.wv
            idx = [self.dictionary[i] for i in range(len(self.dictionary))]
            wv = wv[idx] # [word_cnt, wv_dim]
            if not sparse:
                # self.wv_tfidf = np.matmul(tf_idf.toarray(), wv)
                wv = sp.csr_matrix(wv)
                self.wv_tfidf = tf_idf * wv # [doc_num, wv_dim]
                self.wv_tfidf = self.wv_tfidf.toarray()
            else:
                wv = sp.csr_matrix(wv)
                self.wv_tfidf = tf_idf * wv # [doc_num, wv_dim]

            epoch = math.ceil(self.doc_num / self.batch_size)
            # for i in range(epoch):
            for i in tqdm(range(epoch), total=epoch, desc='BigBird Calculating'):
                down = i * self.batch_size
                up = (i + 1) * self.batch_size if (i + 1) * self.batch_size <= self.doc_num else self.doc_num
                batch_tokens_pt = tokens_pt.copy()
                for key in batch_tokens_pt.data.keys():
                    batch_tokens_pt.data[key] = batch_tokens_pt.data[key][down : up].cuda()
                
                with torch.no_grad():
                    output = self.big_bird_model.forward(**batch_tokens_pt)
                    self.big_bird_rp[down : up, :] = output.pooler_output.cpu().numpy()
                    del output, batch_tokens_pt
                    cuda.empty_cache()

        pass
    
    def _ner_result_extractor(self, result):
        return [(ent.text, ent.type) for sent in result.sentences for ent in sent.ents]

    def build_entity_tfidf(self, doc_entities:List[list]):
        doc_entities_str = [[re.sub(r'\\', '',str(entity)) for entity in entities] for entities in doc_entities]
        vocab = corpora.Dictionary(doc_entities_str)

        doc_entities_encode = [[str(vocab.token2id[entity]) for entity in entities] for entities in doc_entities_str]
        doc_entities_encode = [' '.join(entities) for entities in doc_entities_encode]

        token2id = {str(i):i for i in range(len(vocab))}

        tfidf_tsf = TfidfVectorizer(
            vocabulary=token2id,
            min_df=0.0,
            max_df=1.0
        )

        entities_tfidf_mat = tfidf_tsf.fit_transform(doc_entities_encode).toarray()
        
        return entities_tfidf_mat, vocab, tfidf_tsf

    def _preproceeding(self, texts:list, stopwords:list):
        """
        预处理输入的list[str] text

        input:
        --------
        texts: list[str]
        stopwords: list[str]

        output:
        --------
        list[str], 得到预处理的文本
        """
        tokenizer = RegexpTokenizer(r'\w+')
        p_stemmer = PorterStemmer()
        def __preproceeding(text:str):
            text = text.lower()
            tokens = tokenizer.tokenize(text)
            tokens = list(filter(lambda token: token not in stopwords, tokens))
            tokens = [p_stemmer.stem(token) for token in tokens]
            return tokens
        
        texts = [__preproceeding(text) for text in texts]
        return texts
    
    def update(self, texts:List[str], labels:List[int], stopwords:list=None, ner=True):
        """
        对输入的文本进行训练
        input:
        --------
        texts: list[str]
        labels: List[int], 每篇文章对应的label, 求结果时每一类的lab可能不同
        stopwords: list[str]

        output:
        --------
        None(self)
        """
        assert len(texts) == len(labels), "label与文章数量不相等"
        if stopwords is None:
            stopwords = get_stop_words('en')

        texts = self._preproceeding(texts, stopwords)
        if self.dictionary is None:
            self._texts = texts
            self._labels = labels
            self.dictionary = corpora.Dictionary(texts)
            self.w2vmodel.build_vocab(texts)
        else:
            self._texts.extend(texts)
            self._labels.extend(labels)
            # print(len(self.dictionary))
            self.dictionary.add_documents(texts)
            # print(len(self.dictionary))
            self.w2vmodel.clear_sims()
            self.w2vmodel.build_vocab(self._texts, update=True)
        
        self.tfidf_tsf.vocabulary = self.dictionary.token2id
        Texts = [' '.join(text) for text in texts]
        if ner:
            if not hasattr(self, 'entities'):
                self.entities = []
            with torch.no_grad():
                for text in tqdm(texts, total=len(texts), desc='calculating NER'):
                    self.entities.append(set(self._ner_result_extractor(self.ner_processer(text))))
        else:
            if not hasattr(self, 'entities'):
                self.entities = [set() for _ in range(len(Texts))]
            else:
                for _ in range(len(Texts)):
                    self.entities.append(set())
        # self.entities = [self._ner_result_extractor(self.ner_processer(text)) for text in Texts]
        # self.entities, self.entities_vocab, self.entities_tfidf_tsf = self.build_entity_tfidf(self.entities)

        tokens_pt = self.big_bird_tokenizer(Texts, return_tensors='pt', padding=True)
        tf_idf = self.tfidf_tsf.fit_transform(Texts)
        
        self.w2vmodel.train(self._texts, total_examples=self.w2vmodel.corpus_count, epochs=self.w2vmodel.iter)
        wv = self.w2vmodel.wv
        idx = [self.dictionary[i] for i in range(len(self.dictionary))]
        wv = wv[idx]
        if not self.sparse:
            # self.wv_tfidf = np.matmul(tf_idf.toarray(), wv)
            wv = sp.csr_matrix(wv)
            self.wv_tfidf = tf_idf * wv
            self.wv_tfidf = self.wv_tfidf.toarray()
        else:
            wv = sp.csr_matrix(wv)
            self.wv_tfidf = tf_idf * wv
        
        big_bird_rp = np.zeros((len(Texts), 768))
        epoch = len(Texts) // self.batch_size + 1
        for i in range(epoch):
            down = i * self.batch_size
            up = (i + 1) * self.batch_size if (i + 1) * self.batch_size <= self.doc_num else self.doc_num
            batch_tokens_pt = tokens_pt.copy()
            for key in batch_tokens_pt.data.keys():
                batch_tokens_pt.data[key] = batch_tokens_pt.data[key][down : up].cuda()
                
            with torch.no_grad():
                output = self.big_bird_model.forward(**batch_tokens_pt)
                big_bird_rp[down : up, :] = output.pooler_output.cpu().numpy()
                del output, batch_tokens_pt
                cuda.empty_cache()
        self.big_bird_rp = np.concatenate([self.big_bird_rp, big_bird_rp], aixs=0)
        
class BERTTextRepresentation(object):
    """
    采用BERT模型的文本表示
    input:
    --------
    ```
    texts: list[str], 每一个str为一篇文本
    stopwords: list, 包含所有的停用词
    labels: List[int], 每篇文章对应的label, 求结果时每一类的lab可能不同
    wv_dim: int, 词向量的维度
    batch_size: int, BERT模型的batchsize
    wv_iter: Word2Vec模型迭代次数, 默认为5
    tf_idf_max_df, tfidf_min_df: float in range[0, 1], 参见tf-idf说明
    wv_min_count: int, W2V中纳入计算的最小词频, 为保证W2V和tf-idf模型匹配, 默认为1, 不建议修改
    jobs: int, W2V训练线程数
    random_seed: int, 随机种子
    stanza_path: str, stanza模型的路径, 默认为'../default/'
    sparse: bool, wv_tfidf的输出是否选择为稀疏矩阵, 默认为False
    ```
  
    components:
    --------
    ```
    wv_tfidf: [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量

    bert_rp: [n_doc, 768]的np.ndarray, BERT模型的pooler_output

    dictionary: class: gensim.corpora.Dictionary, 经过预处理后的文本词典

    entities: List[Set], 每篇文章的命名实体集合
    ```
    """
    def __init__(
        self,
        texts: list = None,
        stopwords: list = None,
        labels:List[int] = None,
        wv_dim: int = 100,
        batch_size: int = 10,
        wv_iter: int = 10,
        tfidf_max_df: float = 1.0,
        tfidf_min_df = 1,
        wv_min_count: int = 1,
        jobs: int = 1,
        random_seed: int = 1,
        stanza_path: str = '../default/', 
        sparse: bool = False,
        ner=True
    ):
        """
        采用BERT模型的文本表示
        input:
        --------
        ```
        texts: list[str], 每一个str为一篇文本
        stopwords: list, 包含所有的停用词
        labels: List[int], 每篇文章对应的label, 求结果时每一类的lab可能不同
        wv_dim: int, 词向量的维度
        batch_size: int, BERT模型的batchsize, 默认为10
        wv_iter: Word2Vec模型迭代次数, 默认为5
        tf_idf_max_df, tfidf_min_df: float in range[0, 1], 参见tf-idf说明
        wv_min_count: int, W2V中纳入计算的最小词频, 为保证W2V和tf-idf模型匹配, 默认为1, 不建议修改
        jobs: int, W2V训练线程数
        random_seed: int, 随机种子
        stanza_path: str, stanza模型的路径, 默认为'../default/'
        sparse: bool, wv_tfidf的输出是否选择为稀疏矩阵, 默认为False
        ```
    
        components:
        --------
        ```
        wv_tfidf: [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量

        bert_rp: [n_doc, 768]的np.ndarray, BERT模型的pooler_output

        dictionary: class: gensim.corpora.Dictionary, 经过预处理后的文本词典

        entities: List[Set], 每篇文章的命名实体集合
        ```
        """
        
        super(BERTTextRepresentation, self).__init__()
        # assert len(texts) == len(labels), "label与文章数量不相等"
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        cuda.manual_seed(random_seed)
        os.environ['PYTHONHASHSEED '] = str(random_seed)
        self.wv_dim = wv_dim
        self.batch_size = batch_size
        self._labels = labels
        self.sparse = sparse
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.bert_model = BertModel.from_pretrained('bert-base-uncased').eval().cuda()

        if stopwords is None:
            stopwords = get_stop_words('en')
        
        if texts is not None:
            texts = self._preproceeding(texts, stopwords)
            self._texts = texts
            self.doc_num = len(texts)
            self.bert_rp = np.zeros((self.doc_num, 768))

            self.dictionary = corpora.Dictionary(texts)
            self.tfidf_tsf = TfidfVectorizer(
                stop_words=stopwords,
                min_df=tfidf_min_df,
                max_df=tfidf_max_df,
                vocabulary=self.dictionary.token2id
            )
            self.w2vmodel = Word2Vec(
                sentences = texts,
                size=wv_dim,
                min_count=wv_min_count,
                iter=wv_iter,
                sg=1, hs=1,
                workers=jobs,
                seed=random_seed
            )
        
        else:
            self._texts = None
            self.dictionary = None
            self.tfidf_tsf = TfidfVectorizer(
                stop_words=stopwords,
                min_df=tfidf_min_df,
                max_df=tfidf_max_df
            )
        
            self.w2vmodel = Word2Vec(
                size=wv_dim,
                min_count=wv_min_count,
                iter=wv_iter,
                sg=1, hs=1,
                workers=jobs,
                seed=random_seed
            )

        self.ner_processer = stanza.Pipeline(
            lang='en', dir=stanza_path, 
            processors='tokenize, ner', 
            use_gpu=True
        )
        
        if texts is not None:
            # max_lenth = np.max([len(text) for text in texts])
            # padding = [['<pad>'] * (max_lenth - len(text)) for text in texts]
            Texts = [' '.join(text) for text in texts]

            if ner:
                self.entities = []
                with torch.no_grad():
                    for text in tqdm(Texts, total=len(Texts), desc='calculating NER'):
                        self.entities.append(self._ner_result_extractor(self.ner_processer(text)))
                # self.entities, self.entities_vocab, self.entities_tfidf_tsf = self.build_entity_tfidf(self.entities)
            else:
                # self.entities = np.random.randn(len(Texts), 256)
                self.entities = [set() for _ in range(len(Texts))]

            tokens_pt = self.bert_tokenizer(Texts, return_tensors='pt', padding=True)

            tf_idf = self.tfidf_tsf.fit_transform(Texts) # [doc_num, word_cnt]
            wv = self.w2vmodel.wv
            idx = [self.dictionary[i] for i in range(len(self.dictionary))]
            wv = wv[idx] # [word_cnt, wv_dim]
            if not sparse:
                # self.wv_tfidf = np.matmul(tf_idf.toarray(), wv)
                wv = sp.csr_matrix(wv)
                self.wv_tfidf = tf_idf * wv # [doc_num, wv_dim]
                self.wv_tfidf = self.wv_tfidf.toarray()
            else:
                wv = sp.csr_matrix(wv)
                self.wv_tfidf = tf_idf * wv # [doc_num, wv_dim]

            epoch = math.ceil(self.doc_num / self.batch_size)
            # for i in range(epoch):
            for i in tqdm(range(epoch), total=epoch, desc='BERT Calculating'):
                down = i * self.batch_size
                up = (i + 1) * self.batch_size if (i + 1) * self.batch_size <= self.doc_num else self.doc_num
                batch_tokens_pt = tokens_pt.copy()
                for key in batch_tokens_pt.data.keys():
                    batch_tokens_pt.data[key] = batch_tokens_pt.data[key][down : up, :512].cuda()
                
                with torch.no_grad():
                    output = self.bert_model.forward(**batch_tokens_pt)
                    self.bert_rp[down : up, :] = output.pooler_output.cpu().numpy()
                    del output, batch_tokens_pt
                    cuda.empty_cache()

        pass
    
    def _ner_result_extractor(self, result):
        return [(ent.text, ent.type) for sent in result.sentences for ent in sent.ents]

    def build_entity_tfidf(self, doc_entities:List[list]):
        doc_entities_str = [[re.sub(r'\\', '',str(entity)) for entity in entities] for entities in doc_entities]
        vocab = corpora.Dictionary(doc_entities_str)

        doc_entities_encode = [[str(vocab.token2id[entity]) for entity in entities] for entities in doc_entities_str]
        doc_entities_encode = [' '.join(entities) for entities in doc_entities_encode]

        token2id = {str(i):i for i in range(len(vocab))}

        tfidf_tsf = TfidfVectorizer(
            vocabulary=token2id,
            min_df=0.0,
            max_df=1.0
        )

        entities_tfidf_mat = tfidf_tsf.fit_transform(doc_entities_encode).toarray()
        
        return entities_tfidf_mat, vocab, tfidf_tsf

    def _preproceeding(self, texts:list, stopwords:list):
        """
        预处理输入的list[str] text

        input:
        --------
        texts: list[str]
        stopwords: list[str]

        output:
        --------
        list[str], 得到预处理的文本
        """
        tokenizer = RegexpTokenizer(r'\w+')
        p_stemmer = PorterStemmer()
        def __preproceeding(text:str):
            text = text.lower()
            tokens = tokenizer.tokenize(text)
            tokens = list(filter(lambda token: token not in stopwords, tokens))
            tokens = [p_stemmer.stem(token) for token in tokens]
            return tokens
        
        texts = [__preproceeding(text) for text in texts]
        return texts
    
    def update(self, texts:List[str], labels:List[int], stopwords:list=None, ner=True):
        """
        对输入的文本进行训练
        input:
        --------
        texts: list[str]
        labels: List[int], 每篇文章对应的label, 求结果时每一类的lab可能不同
        stopwords: list[str]

        output:
        --------
        None(self)
        """
        assert len(texts) == len(labels), "label与文章数量不相等"
        if stopwords is None:
            stopwords = get_stop_words('en')

        texts = self._preproceeding(texts, stopwords)
        if self.dictionary is None:
            self._texts = texts
            self._labels = labels
            self.dictionary = corpora.Dictionary(texts)
            self.w2vmodel.build_vocab(texts)
        else:
            self._texts.extend(texts)
            self._labels.extend(labels)
            # print(len(self.dictionary))
            self.dictionary.add_documents(texts)
            # print(len(self.dictionary))
            self.w2vmodel.clear_sims()
            self.w2vmodel.build_vocab(self._texts, update=True)
        
        self.tfidf_tsf.vocabulary = self.dictionary.token2id
        Texts = [' '.join(text) for text in texts]
        if ner:
            if not hasattr(self, 'entities'):
                self.entities = []
            with torch.no_grad():
                for text in tqdm(texts, total=len(texts), desc='calculating NER'):
                    self.entities.append(set(self._ner_result_extractor(self.ner_processer(text))))
        else:
            if not hasattr(self, 'entities'):
                self.entities = [set() for _ in range(len(Texts))]
            else:
                for _ in range(len(Texts)):
                    self.entities.append(set())
            
        # self.entities = [self._ner_result_extractor(self.ner_processer(text)) for text in Texts]
        # self.entities, self.entities_vocab, self.entities_tfidf_tsf = self.build_entity_tfidf(self.entities)

        tokens_pt = self.bert_tokenizer(Texts, return_tensors='pt', padding=True)
        tf_idf = self.tfidf_tsf.fit_transform(Texts)
        
        self.w2vmodel.train(self._texts, total_examples=self.w2vmodel.corpus_count, epochs=self.w2vmodel.iter)
        wv = self.w2vmodel.wv
        idx = [self.dictionary[i] for i in range(len(self.dictionary))]
        wv = wv[idx]
        if not self.sparse:
            # self.wv_tfidf = np.matmul(tf_idf.toarray(), wv)
            wv = sp.csr_matrix(wv)
            self.wv_tfidf = tf_idf * wv
            self.wv_tfidf = self.wv_tfidf.toarray()
        else:
            wv = sp.csr_matrix(wv)
            self.wv_tfidf = tf_idf * wv
        
        bert_rp = np.zeros((len(Texts), 768))
        epoch = len(Texts) // self.batch_size + 1
        for i in range(epoch):
            down = i * self.batch_size
            up = (i + 1) * self.batch_size if (i + 1) * self.batch_size <= self.doc_num else self.doc_num
            batch_tokens_pt = tokens_pt.copy()
            for key in batch_tokens_pt.data.keys():
                batch_tokens_pt.data[key] = batch_tokens_pt.data[key][down : up, :512].cuda()
                
            with torch.no_grad():
                output = self.bert_model.forward(**batch_tokens_pt)
                bert_rp[down : up, :] = output.pooler_output.cpu().numpy()
                del output, batch_tokens_pt
                cuda.empty_cache()
        self.bert_rp = np.concatenate([self.bert_rp, bert_rp], aixs=0)
        
       
