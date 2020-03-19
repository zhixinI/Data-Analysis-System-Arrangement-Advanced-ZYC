# Day03

### 1.subplots

创建子图

使用方法：subplot（m,n,p）或者subplot（m  n  p）。

subplot是将多个图画到一个平面上的工具。其中，m表示是图排成m行，n表示图排成n列，也就是整个figure中有n个图是排成一行的，一共m行，如果m=2就是表示2行图。p表示图所在的位置，p=1表示从左到右从上到下的第一个位置。

```python
subplot(2,2,1); % 2、2、1之间没有空格也可以
%在第一块绘图
subplot(2,2,2);
%在第二块绘图
subplot(2,2,3);
%在第三块绘图
subplot(2,2,4);
%在第四块绘图
```

### 2. 盒形图boxplot()

形图又称箱图，主要用来显示与类别相关的数据分布。

```python
seaborn.boxplot(x=None, y=None, hue=None, 
                data=None, order=None, hue_order=None, 
                orient=None, color=None, palette=None, 
                saturation=0.75, width=0.8, dodge=True, 
                fliersize=5, linewidth=None, whis=1.5, 
                notch=False, ax=None, **kwargs)
```

```markdown
x,y,hue:数据字段变量名(如上表，date,name,age,sex为数据字段变量名)
	作用：根据实际数据，x,y常用来指定x,y轴的分类名称，
hue常用来指定第二次分类的数据类别(用颜色区分)
data: DataFrame,数组或数组列表
order,hue_order:字符串列表
	作用：显式指定分类顺序，eg. order=[字段变量名1，字段变量名2,...]
orient:方向：v或者h
	作用：设置图的绘制方向(垂直或水平)，如何选择：一般是根据输入变量的数据类型(dtype)推断出来。
color:matplotlib 颜色
palette:调色板名称，list类别或者字典
	作用：用于对数据不同分类进行颜色区别
saturation 饱和度：float
dodge:bool
	作用：若设置为True则沿着分类轴，将数据分离出来成为不同色调级别的条带，否则，每个级别的点将相互叠加
size:float
	作用：设置标记大小(标记直径，以磅为单位)
edgecolor:matplotlib color，gray
	作用：设置每个点的周围线条颜色
linewidth:float
	作用：设置构图元素的线宽度
```

[更多案例参考](https://cloud.tencent.com/developer/article/1517202)

### 3. pandas.DataFrame.select_dtypes

`DataFrame.``select_dtypes`（*self*，*include = None*，*exclude = None* ） →'DataFrame' [[源代码）](http://github.com/pandas-dev/pandas/blob/v1.0.3/pandas/core/frame.py#L3348-L3476)

DataFrame的select_dtypes方法返回具有指定数据类型的列。

```python
DataFrame.select_dtypes(self[, include, exclude])
```

```markdown
参数：
include：要返回的列的数据类型，标量或列表
exclude：要排除的列的数据类型，标量或列表
返回值：
select_dtypes()方法返回原数据帧的子集，由include中声明的 列组成，并且排除exclude中声明的列。
```

```python
>>> df = pd.DataFrame({'a': [1, 2] * 3,
...                    'b': [True, False] * 3,
...                    'c': [1.0, 2.0] * 3})
>>> df
        a      b  c
0       1   True  1.0
1       2  False  2.0
2       1   True  1.0
3       2  False  2.0
4       1   True  1.0
5       2  False  2.0
```

[更多细致参考链接](http://cw.hubwiz.com/card/c/pandas-manual/1/5/8/)

### 4. GradientBoostingRegressor

python中的scikit-learn包提供了很方便的GradientBoostingRegressor和GBDT的函数接口，可以很方便的调用函数就可以完成模型的训练和预测GradientBoostingRegressor函数的参数如下：

```
class sklearn.ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')[source]¶
```

loss: 选择损失函数，默认值为ls(least squres)

learning_rate: 学习率，模型是0.1

n_estimators: 弱学习器的数目，默认值100

max_depth: 每一个学习器的最大深度，限制回归树的节点数目，默认为3

min_samples_split: 可以划分为内部节点的最小样本数，默认为2

min_samples_leaf: 叶节点所需的最小样本数，默认为1

[更多细致参考](https://www.cnblogs.com/zhubinwang/p/5170087.html)

### 5. GridSearchCV

GridSearchCV，它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数。但是这个方法适合于小数据集，一旦数据的量级上去了，很难得出结果。这个时候就是需要动脑筋了。数据量比较大的时候可以使用一个快速调优的方法——坐标下降。它其实是一种贪心算法：拿当前对模型影响最大的参数调优，直到最优化；再拿下一个影响最大的参数调优，如此下去，直到所有的参数调整完毕。这个方法的缺点就是可能会调到局部最优而不是全局最优，但是省时间省力，巨大的优势面前，还是试一试吧，后续可以再拿bagging再优化。

通常算法不够好，需要调试参数时必不可少。比如SVM的惩罚因子C，核函数kernel，gamma参数等，对于不同的数据使用不同的参数，结果效果可能差1-5个点，sklearn为我们提供专门调试参数的函数grid_search。

*class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch=‘2\*n_jobs’, error_score=’raise’, return_train_score=’warn’)*

|                                    |                                                              |
| ---------------------------------- | ------------------------------------------------------------ |
| ***estimator\***                   | 选择使用的分类器，并且传入除需要确定最佳的参数之外的其他参数。每一个分类器都需要一个scoring参数，或者score方法：estimator=RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt',random_state=10) |
| ***param_grid\***                  | 需要最优化的参数的取值，值为字典或者列表，例如：param_grid =param_test1，param_test1 = {'n_estimators':range(10,71,10)}。 |
| ***scoring=None\***                | 模型评价标准，默认None,这时需要使用score函数；或者如scoring='roc_auc'，根据所选模型不同，评价准则不同。字符串（函数名），或是可调用对象，需要其函数签名形如：scorer(estimator, X, y)；如果是None，则使用estimator的误差估计函数。具体值的选取看本篇第三节内容。 |
| ***fit_params=None\***             |                                                              |
| ***n_jobs=1\***                    | n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值        |
| ***iid=True\***                    | **iid**:默认True,为True时，默认为各个样本fold概率分布一致，误差估计为所有样本之和，而非各个fold的平均。 |
| ***refit=True\***                  | 默认为True,程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行，作为最终用于性能评估的最佳模型参数。即在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集。 |
| ***cv=None\***                     | 交叉验证参数，默认None，使用三折交叉验证。指定fold数量，默认为3，也可以是yield训练/测试数据的生成器。 |
| ***verbose=0\*, \*scoring=None\*** | **verbose**：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。 |
| ***pre_dispatch=‘2\*n_jobs’\***    | 指定总共分发的并行任务数。当n_jobs大于1时，数据将在每个运行点进行复制，这可能导致OOM，而设置pre_dispatch参数，则可以预先划分总共的job数量，使数据最多被复制pre_dispatch次 |
| ***error_score=’raise’\***         |                                                              |
| ***return_train_score=’warn’\***   | 如果“False”，cv_results_属性将不包括训练分数                 |

[更多细致参考](https://blog.csdn.net/weixin_41988628/article/details/83098130)





























