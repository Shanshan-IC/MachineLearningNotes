PCA的目的是降维。

降维的目的：

1.减少预测变量的个数

2.确保这些变量是相互独立的

3.提供一个框架来解释结果

实现步骤：

1.原始数据A特征中心化B。即每一维的数据都减去该维的均值

2.计算B的协方差矩阵C

3.计算矩阵C的特征根和特征向量

4.选取大的特征值对应的特征向量，得到新的数据集，做矩阵相乘
