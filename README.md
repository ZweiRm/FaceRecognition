# 人脸识别
本项目基于 Hong Kong Baptist University COMP 7930 Big Data Analytics Mini Project.  
使用 TensorFlow 对三个人脸数据库做分类：ORL, Yale 和 YaleB.  
对于每个数据库，均实现三个模型来进行识别，分别是：Softmax, Soft max with PCA, CNN.  

在之前的项目里，九个模型的准确度分别是：  
||Yale|ORL|YaleB|
|:--:|:--:|:--:|:--:|
|CNN|84.85%|91.25%|96.27%|
|Softmax|76%|82.25%|93.79%|
|PCA Softmax|78.79%|91.25%|79.02%|