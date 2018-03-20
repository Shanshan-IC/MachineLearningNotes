## 三种方法求解线性回归

1. **Normal Equation**

<img src="http://latex.codecogs.com/gif.latex?\theta =(X^TX)^{-1}X^Ty" />

优点：简单方便，不需要做feature scaling，特征较少的时候适用。

缺点：求逆运算时，对机器容量要求高

2. **Stochastic Gradient Descent**

用随机梯度下降法求解权重

