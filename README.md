# TR_Cluster代码结构

## BigBirddemo

试验BigBird模型, 目前BigBird模型来自transformers==4.5.0dev0版本，在项目开始时尚未正式发布

## demo

文本表示+话题聚类示例代码+评估

## hac

层次聚类算法

## singlepass

single-pass算法

## textrepresentation

文本表示
包含用LDA+W2V&tf-idf、BigBird+W2V&tf-idf和最终采用的BERT+tf-idf&W2V的文本向量表示方法

## label

打标数据的算法

## test_ner

采用stanza库的NER测试代码

## classifier

分类器代码

# 环境

操作系统Win10 21H4
环境 VScode1.57
python 3.7.4
python包版本要求:

```
scipy==1.4.1
sklearn==0.21.3
gensim==3.8.3
typesentry==0.2.7
stop_words==2018.7.23
nltk==3.4.5
numpy==1.19.5
torch==1.7.1
transformers>=4.5.0
stanza==1.2
pandas==0.25.1
```
