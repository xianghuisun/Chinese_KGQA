# Chinese_KGQA
**该仓库目的是实现基于知识图谱的中文问答系统**

- code文件夹下存放所有的问答模型，每一个模型是一个文件夹，用模型的名称命名
- data文件夹下存放问答数据集
- kg文件夹下存放知识图谱
- png文件夹下存放所需要的图片
- result文件夹下记录code文件夹下的每一个模型在data文件夹下的每一个数据集上的实验结果



## code文件夹

该文件夹下的子文件夹都是以模型名称命名，目前包括：

- [TransferNet](https://github.com/shijx12/TransferNet)
- [Embed-KGQA](https://github.com/malllabiisc/EmbedKGQA)

## data文件夹

该文件夹下的子文件夹都是以QA数据集名称命名，目前包括：

- WebQSP
- nlpcc2018(官网：http://tcci.ccf.org.cn/conference/2018/taskdata.php，选择task7，Open Domain Question Answering，即可下载数据集。数据集包含知识图谱和问答数据)

## kg文件夹

该文件夹下的子文件是以知识图谱的名称命名，目前包括：

- fbwq_full（来源https://drive.google.com/file/d/1uWaavrpKKllVSQ73TTuLWPc4aqVvrkpx/view?usp=sharing，也就是[Embed-KGQA](https://github.com/malllabiisc/EmbedKGQA)给出的整理好的知识图谱）
- nlpcc2018（来源就是官网http://tcci.ccf.org.cn/conference/2018/taskdata.php中下载的知识图谱）
- PKU（北大的中文百科知识图谱。链接：https://pan.baidu.com/s/1Br8eU60t2fV4crtC2HOlSg 提取码：tvv1）





## 相关工作和资源总结



### 博客

1. [开放知识图谱](https://blog.csdn.net/TgqDT3gGaMdkHasLZv)



### GitHub链接

#### 关于KGE

| 链接                                                 | 说明                                             | 备注                                                         |
| ---------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| https://github.com/uma-pi1/kge/                      | 专门用于KGE的仓库，实现了诸多KGE模型             | 代码看的有点懵                                               |
| https://github.com/Sujit-O/pykg2vec                  | 另一个专门用于KGE的仓库，实现了诸多KGE模型       | 目前还不清楚和上个kge仓库哪个好用                            |
| https://github.com/facebookresearch/PyTorch-BigGraph | 专门针对超大规模知识图谱嵌入、存储以及应用的仓库 |                                                              |
| https://github.com/facebookresearch/kbc              | BigGraph中的一个子模块，专门实现了ComplEx模型    | 代码量简洁，很方便的实现ComplEx模型在FB15k和其他数据集的实验 |

#### 关于KGQA

| 链接                                      | 说明                           | 备注 |
| ----------------------------------------- | ------------------------------ | ---- |
| https://github.com/malllabiisc/EmbedKGQA/ | ACL2020论文EmbedKGQA的源码     |      |
| https://github.com/shijx12/TransferNet    | EMNLP2021论文TransferNet的源码 |      |

#### 其它

| 链接                                          | 说明                                                         | 备注                                       |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| https://github.com/BDBC-KG-NLP/QA-Survey      | 北航某个团队的项目，对问答系统的总结，包括各种任务问答（文本、图谱等）。 | 总结的相当全面，包括学术界和工业界         |
| https://github.com/liuhuanyong/QAonMilitaryKG | 360某个NLP专家的代码，军事领域知识图谱问答系统。             | 这个专家的GitHub仓库很多都是关于知识图谱的 |
| https://github.com/BshoterJ/awesome-kgqa      | 该仓库主要记录了关于KGQA的一些资源，包括论文、比赛等         |                                            |

### 论文

#### 综述

- [Knowledge Graph Embedding: A Survey of Approaches and Applications](https://persagen.com/files/misc/Wang2017Knowledge.pdf)
- [A survey of embedding models of entities and relationships for knowledge graph completion](https://arxiv.org/pdf/1703.08098.pdf)

#### KGQA

- [Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings](https://aclanthology.org/2020.acl-main.412.pdf)
- [TransferNet: An Effective and Transparent Framework for Multi-hop Question Answering over Relation Graph](https://arxiv.org/abs/2104.07302)

#### KGE



### 中文开放域知识图谱

- PKU（北大的中文百科知识图谱。链接：https://pan.baidu.com/s/1Br8eU60t2fV4crtC2HOlSg 提取码：tvv1）
- nlpcc2018(官网：http://tcci.ccf.org.cn/conference/2018/taskdata.php 选择task7 Open Domain Question Answering，即可下载数据集。数据集包含知识图谱和问答数据)
- 思知知识图谱（地址：https://www.ownthink.com/docs/kg/）

这三个是比较大的开放域知识图谱，此外还有一些垂类领域如军事、医学、法律等领域的知识图谱，这里不再介绍。

### 中文问答数据（基于开放域知识图谱）

- ccks（https://github.com/pkumod/CKBQA/tree/master/data）
- nlpcc2018 (官网：http://tcci.ccf.org.cn/conference/2018/taskdata.php 选择task7 Open Domain Question Answering，即可下载数据集。数据集包含知识图谱和问答数据)

### 个人总结的博客

尽量按照所列顺序阅读

1. [简要总结一篇知识图谱嵌入综述](https://blog.csdn.net/m0_45478865/article/details/121304792)
2. [简单实现几篇知识图谱嵌入(Knowledge Graph Embedding，KGE)模型](https://blog.csdn.net/m0_45478865/article/details/121195480)
3. [KGQA概览](https://blog.csdn.net/m0_45478865/article/details/121104817)
4. [EmbedKGQA论文简要解读](https://blog.csdn.net/m0_45478865/article/details/121203874)
