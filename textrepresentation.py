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

from transformers.models import big_bird
warnings.filterwarnings('ignore')
from transformers import BigBirdTokenizer, BigBirdModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.cuda as cuda
import math
import typesentry
tc1 = typesentry.Config()
Isinstance = tc1.is_type


class TextRepresentation(object):
    """
    文本表示的类
    input:
    --------
    ```
    texts: list[str], 每一个str为一篇文本
    stopwords: list, 包含所有的停用词
    wv_dim: int, 词向量的维度
    lda_n_compontents: int, LDA模型的主题数目
    wv_iter: Word2Vec模型迭代次数, 默认为5
    lda_max_iter: LDA模型迭代次数, 默认为10
    tf_idf_max_df, tfidf_min_df: float in range[0, 1], 参见tf-idf说明
    wv_min_count: int, W2V中纳入计算的最小词频, 为保证W2V和tf-idf模型匹配, 默认为1, 不建议修改
    jobs: int, 训练线程数
    random_seed: int, 随机种子
    sparse: bool, wv_tfidf是否选择为稀疏矩阵, 默认为False
    ```
  
    output:
    --------
    ```
    wv_tfidf: [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量

    lda_doc_topic: [n_doc, lda_n_components]的np.ndarray, LDA训练的文档-主题矩阵
        
    lda_topic_term: [lda_n_components, vocab_size]的np.ndarray, LDA训练得到的主题-词矩阵

    dictionary: class: gensim.corpora.Dictionary, 经过预处理后的文本词典
    ```
    """
    def __init__(
        self,
        texts: list = None,
        stopwords: list = None,
        wv_dim: int = 100,
        lda_n_components: int = 100,
        wv_iter: int = 5,
        lda_max_iter: int = 10,
        tfidf_max_df: float = 1.0,
        tfidf_min_df = 1,
        wv_min_count: int = 1,
        jobs: int = 12,
        random_seed: int = 0,
        sparse: bool = False
    ):
        """
        input:
        --------
        ```
        texts: list[str], 每一个str为一篇文本
        stopwords: list, 包含所有的停用词
        wv_dim: int, 词向量的维度
        lda_n_compontents: int, LDA模型的主题数目
        wv_iter: Word2Vec模型迭代次数, 默认为5
        lda_max_iter: LDA模型迭代次数, 默认为10
        tf_idf_max_df, tfidf_min_df: float in range[0, 1], 参见tf-idf说明
        wv_min_count: int, W2V中纳入计算的最小词频, 为保证W2V和tf-idf模型匹配, 默认为1, 不建议修改
        jobs: int, 训练线程数
        random_seed: int, 随机种子
        sparse: bool, wv_tfidf是否选择为稀疏矩阵, 默认为False
        ```
  
        output:
        --------
        ```
        wv_tfidf: [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量

        lda_doc_topic: [n_doc, lda_n_components]的np.ndarray, LDA训练的文档-主题矩阵
        
        lda_topic_term: [lda_n_components, vocab_size]的np.ndarray, LDA训练得到的主题-词矩阵

        dictionary: class: gensim.corpora.Dictionary, 经过预处理后的文本词典
        ```
        """
        super(TextRepresentation, self).__init__()
        self.wv_dim = wv_dim
        self.lda_n_component = lda_n_components
        self.sparse = sparse
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
                workers=jobs
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
                workers=jobs
            )
        self.lda_model = LatentDirichletAllocation(
            n_components=lda_n_components,
            learning_method='batch', 
            n_jobs=jobs,
            max_iter=lda_max_iter,
            random_state=random_seed
        )
        if texts is not None:
            texts = [' '.join(text) for text in texts]
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
    
    def update(self, texts:list, stopwords:list=None):
        """
        对输入的文本进行训练
        input:
        --------
        texts: list[str]
        stopwords: list[str]

        output:
        --------
        None(self)
        """
        if stopwords is None:
            stopwords = get_stop_words('en')

        texts = self._preproceeding(texts, stopwords)
        if self.dictionary is None:
            self._texts = texts
            self.dictionary = corpora.Dictionary(texts)
            self.w2vmodel.build_vocab(texts)
        else:
            self._texts.extend(texts)
            # print(len(self.dictionary))
            self.dictionary.add_documents(texts)
            # print(len(self.dictionary))
            self.w2vmodel.clear_sims()
            self.w2vmodel.build_vocab(self._texts, update=True)
        self.tfidf_tsf.vocabulary = self.dictionary.token2id
        Texts = [' '.join(text) for text in self._texts]
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
    wv_dim: int, 词向量的维度
    batch_size: int, BigBird模型的batchsize
    wv_iter: Word2Vec模型迭代次数, 默认为5
    tf_idf_max_df, tfidf_min_df: float in range[0, 1], 参见tf-idf说明
    wv_min_count: int, W2V中纳入计算的最小词频, 为保证W2V和tf-idf模型匹配, 默认为1, 不建议修改
    jobs: int, W2V训练线程数
    random_seed: int, 随机种子
    sparse: bool, wv_tfidf的输出是否选择为稀疏矩阵, 默认为False
    ```
  
    components:
    --------
    ```
    wv_tfidf: [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量

    big_bird_rp: [n_doc, 768]的np.ndarray, BigBird模型的pooler_output

    dictionary: class: gensim.corpora.Dictionary, 经过预处理后的文本词典
    ```
    """
    def __init__(
        self,
        texts: list = None,
        stopwords: list = None,
        wv_dim: int = 100,
        batch_size: int = 10,
        wv_iter: int = 5,
        tfidf_max_df: float = 1.0,
        tfidf_min_df = 1,
        wv_min_count: int = 1,
        jobs: int = 12,
        random_seed: int = 0,
        sparse: bool = False
    ):
        """
        采用BigBird模型的文本表示
        input:
        --------
        ```
        texts: list[str], 每一个str为一篇文本
        stopwords: list, 包含所有的停用词
        wv_dim: int, 词向量的维度
        batch_size: int, BigBird模型的batchsize, 默认为10
        wv_iter: Word2Vec模型迭代次数, 默认为5
        tf_idf_max_df, tfidf_min_df: float in range[0, 1], 参见tf-idf说明
        wv_min_count: int, W2V中纳入计算的最小词频, 为保证W2V和tf-idf模型匹配, 默认为1, 不建议修改
        jobs: int, W2V训练线程数
        random_seed: int, 随机种子
        sparse: bool, wv_tfidf的输出是否选择为稀疏矩阵, 默认为False
        ```
    
        components:
        --------
        ```
        wv_tfidf: [n_doc, wv_dim]的np.ndarray, 得到tf-idf加权融合的词向量

        big_bird_rp: [n_doc, 768]的np.ndarray, BigBird模型的pooler_output

        dictionary: class: gensim.corpora.Dictionary, 经过预处理后的文本词典
        ```
        """
        
        super(BigBirdTextRepresentation, self).__init__()
        self.wv_dim = wv_dim
        self.batch_size = batch_size
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
                workers=jobs
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
                workers=jobs
            )
        
        if texts is not None:
            # max_lenth = np.max([len(text) for text in texts])
            # padding = [['<pad>'] * (max_lenth - len(text)) for text in texts]
            Texts = [' '.join(text) for text in texts]
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
            for i in range(epoch):
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
    
    def update(self, texts:list, stopwords:list=None):
        """
        对输入的文本进行训练
        input:
        --------
        texts: list[str]
        stopwords: list[str]

        output:
        --------
        None(self)
        """
        if stopwords is None:
            stopwords = get_stop_words('en')

        texts = self._preproceeding(texts, stopwords)
        if self.dictionary is None:
            self._texts = texts
            self.dictionary = corpora.Dictionary(texts)
            self.w2vmodel.build_vocab(texts)
        else:
            self._texts.extend(texts)
            # print(len(self.dictionary))
            self.dictionary.add_documents(texts)
            # print(len(self.dictionary))
            self.w2vmodel.clear_sims()
            self.w2vmodel.build_vocab(self._texts, update=True)
        
        self.tfidf_tsf.vocabulary = self.dictionary.token2id
        Texts = [' '.join(text) for text in texts]
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
        



        
       
