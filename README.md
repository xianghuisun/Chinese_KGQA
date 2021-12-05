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



