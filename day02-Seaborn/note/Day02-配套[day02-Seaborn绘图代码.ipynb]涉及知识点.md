### Day02

### 一、Seaborn的介绍

- Seaborn是基于[matplotlib](https://matplotlib.org/)的Python数据可视化库。Seaborn的底层是基于Matplotlib的，它提供了一种高度交互式界面，它提供了用于绘制引人入胜且内容丰富的统计图形的高级界面。

#### 1.1 Seaborn vs Matplotlib

- 点餐为例，Matplotlib是独立点菜，Seaborn是点套餐；

- Seaborn是用户把自己常用到的可视化绘图过程进行了函数封装，形成的一个“快捷方式”，它相比Matplotlib的好处是代码更简洁，可以用一行代码实现一个清晰好看的可视化输出。主要的缺点则是定制化能力会比较差，只能实现固化的一些可视化模板类型；

  Matplotlib是可以实现高度定制化绘图的，高度定制化可以让你获得最符合心意的可视化输出结果，但也因此需要设置更多的参数，因而代码更加复杂一些；

- 两者相互补充，Seaborn，一般来说基本是够用的，而且特别容易上手；但很多时候Seaborn输出的结果确实可能不如人意，这时候需要结合Matplotlib做些修改，就需要深入学习下Matplotlib；

- Seaborn是在matplotlib的基础上进行了更高级的API封装，seaborn能做出很具有吸引力的图，而使用matplotlib就能制作具有更多特色的图。应该把Seaborn视为matplotlib的补充，而不是替代物。同时它能高度兼容numpy与pandas数据结构以及scipy与statsmodels等统计模式；
- Matplotlib的不足：图表不够美观；数据分析的时候需要的条件太多；和pandas的结合一般；

- Seaborn的优势：可以绘制高颜值的图表；专门为数据分析设计的可视化库；对于pandas的兼容兼容性非常好。

#### 1.2 图表的构成

1. 关联图：散布图scatterplot、线图lineplot、分面网格relplot
2. 分类图：柱状图barplot、箱图boxplot、小提琴图violinplot、散布图(stripplot、swarmplot)、分面网格catplot
3. 分布图：单变量分布图distplot、核密度图kdeplot
4. 回归图：线性回归图regplot、分面网格线性回归图lmplot
5. 矩阵图：热力图heatmap、聚类图clustermap




### 二、Seaborn的内置数据集

`sns.load_dataset()`函数可以加载内部的数据集,返回一个DataFrame对象：

sns.load_dataset(name,cache=True,data_home,**kwages)

| 参数      | 类型    | 含义                                                         |
| :-------- | :------ | ------------------------------------------------------------ |
| name      | string  | 是数据集的名称                                               |
| cache     | boolean | 可选是否缓存；如果为True，则在本地缓存数据并在后续调用中使用缓存。 |
| data_home | string  | 用于存储缓存数据的目录。 默认情况下使用 ~/seaborn-data/      |
| **kwages  | dict    | 传递给 pandas.read_csv                                       |
| 数据集    | -       | https://github.com/mwaskom/seaborn-data                      |
| 参考链接  | -       | https://www.cntofu.com/book/172/docs/70.md                   |

### 三、Seaborn的风格设置

#### 3.1 set_style()

```py
seaborn.set_style(style=None, rc=None)
```

| 参数     | 类型 | 含义                                                         |
| -------- | ---- | ------------------------------------------------------------ |
| style    | dict | or one of {darkgrid, whitegrid, dark, white, ticks}选择这里面的其中一个；<br />darkgrid(暗网格背景),whitegrid(白网格背景),dark(暗色调背景),white(白色背景),ticks(带有刻度的背景) |
| rc       | Dict | Parameter mappings to override the values in the preset seaborn style dictionaries. This only updates parameters that are considered part of the style definition. 参数映射将覆盖预设的seaborn样式字典中的值。 这只会更新被认为是样式定义一部分的参数。 |
| 参考链接 |      | https://www.cntofu.com/book/172/docs/47.md                   |

例子：

```python
>>> set_style("whitegrid")
>>> set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
```

#### 3.2 numpy.log(底数计算,10和e常见)

```python
numpy.log(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'log'>¶
```

例子：

```python
import numpy as np
#以10为底；
>>> np.log10(x)
#以e为底，e不需要写出来，以4为底的对数
>>> np.log(4)
```

[更多参考：numpy.log官方网站](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html)

#### 3.3 pandas.DataFrame.from_dict

*classmethod* `DataFrame.``from_dict`**（***data*，*orient ='columns'*，*dtype = None*，*columns = None* **）**

从类似数组或字典的字典构造DataFrame；

| 参数                        | 类型 | 含义                                                         |
| --------------------------- | ---- | ------------------------------------------------------------ |
| data                        | dict | 数据；格式为{field：array-like}或{field：dict}。             |
| orient:{‘columns’，‘index’} | -    | 默认’列’数据的“方向”。如果传递的dict的键应该是结果DataFrame的列，则传递’columns’（默认值）。否则，如果键应该是行，则传递’index’。 |
| dtype                       |      | 数据类型                                                     |
| columns                     | list | 时使用的列标签`orient='index'`。如果与一起使用，则会引发ValueError `orient='columns'`。 |
| 更多参考：                  |      | https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.from_dict.html |

例子：

```python
>>> data = {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}
>>> pd.DataFrame.from_dict(data)
   col_1 col_2
0      3     a
1      2     b
2      1     c
3      0     d
#指定orient='index'使用字典键作为行来创建DataFrame：
>>> data = {'row_1': [3, 2, 1, 0], 'row_2': ['a', 'b', 'c', 'd']}
>>> pd.DataFrame.from_dict(data, orient='index')
       0  1  2  3
row_1  3  2  1  0
row_2  a  b  c  d
#使用“索引”方向时，可以手动指定列名称：
>>> pd.DataFrame.from_dict(data, orient='index',
                       columns=['A', 'B', 'C', 'D'])
       A  B  C  D
row_1  3  2  1  0
row_2  a  b  c  d
```

#### 3.4 plt.subplots()

在当前图形上添加一个子图[[matplotlib.pyplot.subplot官方链接](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot )]

返回一个包含figure和axes对象的元组,`fig,ax = plt.subplots()`将元组分解为fig和ax两个变量。;

它是用来创建 总画布/figure“窗口”的，有figure就可以在上边（或其中一个子网格/subplot上）作图了，（fig：是figure的缩写）。

- plt.subplot(111)是plt.subplot(1, 1, 1)另一个写法而已，更完整的写法是plt.subplot(nrows=1, ncols=1, index=1);

- fig, ax = plt.subplots()等价于fig, ax = plt.subplots(11);
- fig, axes = plt.subplots(23)：即表示一次性在figure上创建成2*3的网格，使用plt.subplot()只能一个一个的添加;[[参考链接](https://blog.csdn.net/htuhxf/article/details/82986440)]

例子：

```python
>>> fig,(ax1,ax2) = plt.subplots(1,2)
画一个1行2列的；
```

#### 3.5 seaborn.scatterplot

[[参考：Seaborn系列 | 散点图scatterplot()](https://cloud.tencent.com/developer/article/1506466)]

```python
#可以通过调整颜色、大小和样式等参数来显示数据之间的关系。
seaborn.scatterplot(x=None, y=None, hue=None,
                    style=None, size=None, data=None, 
                    palette=None, hue_order=None, hue_norm=None,
                    sizes=None, size_order=None, size_norm=None, 
                    markers=True, style_order=None, x_bins=None,
                    y_bins=None, units=None, estimator=None, 
                    ci=95, n_boot=1000, alpha='auto', x_jitter=None,
                    y_jitter=None, legend='brief', ax=None, **kwargs)
```

| 参数      | 含义/类型                | 作用                                                         |
| --------- | ------------------------ | ------------------------------------------------------------ |
| x ,y      | 数据中变量的名称;        | 对将生成具有不同颜色的元素的变量进行分组。可以是分类或数字;  |
| size      | 数据中的名称;            | 根据指定的名称(列名)，根据该列中的数据值的大小生成具有不同大小的效果。可以是分类或数字。 |
| style     | 数据中变量名称           | 对将生成具有不同破折号、或其他标记的变量进行分组。           |
| palette   | 调试板名称(list，或dict) | 设置hue指定的变量的不同级别颜色。                            |
| hue_order | 列表(list)类型           | 指定hue变量出现的指定顺序，否则他们是根据数据确定的。        |
| hue_norm  | -                        | tuple或Normalize对象                                         |
| sizes     | list dict或tuple类型     | 设置线宽度，当其为数字时，它也可以是一个元组，指定要使用的最大和最小值，会自动在该范围内对其他值进行规范化。 |

#### 3.6 seaborn.despine

边框控制[[Seaborn0.9中文文档seaborn.despine](https://www.cntofu.com/book/172/docs/71.md)]

```python
sns.despine(fig=None,ax=None,top=True,right=True,left=False,bottom=False,offset=None,trim=False,)
```

| 参数   | 含义/类型            | 作用                                                         |
| ------ | -------------------- | ------------------------------------------------------------ |
| fig    | matplotlib值，可选； | 从图中移除顶部和右侧脊柱。                                   |
| ax     | matplotlib 轴, 可选  | 去除所有轴脊柱，默认使用当前数值。                           |
| top    | boolean, 可选        | 去除顶部特定的轴脊柱。如果为 True，去除脊柱。                |
| right  | boolean, 可选        | 去除右侧特定的轴脊柱。如果为 True，去除脊柱。                |
| left   | boolean, 可选        | 去除左侧特定的轴脊柱。如果为 True，去除脊柱。                |
| bottom | boolean, 可选        | 去除底部特定的轴脊柱。如果为 True，去除脊柱。                |
| offset | int or dict, 可选    | 绝对距离（以磅为单位）应将脊椎移离轴线（负值向内移动脊柱）。 单个值适用于所有脊柱; 字典可用于设置每侧的偏移值。 |
| trim   | bool, 可选           | 如果为True，则将脊柱限制为每个非去除脊柱的轴上的最小和最大主刻度。 |

#### 3.7 seaborn.lineplot()折线图

```python
seaborn.lineplot(x=None, y=None, hue=None, 
                 size=None, style=None, data=None,
                 palette=None, hue_order=None, hue_norm=None, 
                 sizes=None, size_order=None, size_norm=None, 
                 dashes=True, markers=None, style_order=None, 
                 units=None, estimator='mean', ci=95, n_boot=1000,
                 sort=True, err_style='band', err_kws=None,
                 legend='brief', ax=None, **kwargs)
```

| 参数      | 属性/含义                              | 作用                                                         |
| --------- | -------------------------------------- | ------------------------------------------------------------ |
| x,y       | -                                      | 数据中变量的名称;                                            |
| hue       | 数据中变量名称(比如：二维数据中的列名) | 对将要生成不同颜色的线进行分组，可以是分类或数据。           |
| size      | 数据中变量名称(比如：二维数据中的列名) | 对将要生成不同宽度的线进行分组，可以是分类或数据。           |
| style     | 数据中变量名称(比如：二维数据中的列名) | 对将生成具有不同破折号、或其他标记的变量进行分组。           |
| palette   | 调试板名称，列表或字典类型             | 设置hue指定的变量的不同级别颜色。                            |
| hue_order | 列表(list)类型                         | 指定hue变量出现的指定顺序，否则他们是根据数据确定的。        |
| hue_norm  | -                                      | tuple或Normalize对象                                         |
| sizes     | list dict或tuple类型                   | 设置线宽度，当其为数字时，它也可以是一个元组，指定要使用的最大和最小值，会自动在该范围内对其他值进行规范化。 |
| units     | -                                      | 对变量识别抽样单位进行分组，使用时，将为每个单元绘制一个单独的行。 |
| estimator | pandas方法的名称或回调函数或者None     | 用于在同一x水平上聚合y变量的多个观察值的方法，如果为None，则将绘制所有观察结果。 |

[折线图lineplot详细案例](https://cloud.tencent.com/developer/article/1506467)

#### 3.8 color_palette(调色板)

在介绍seaborn的配色使用之前，首先需要介绍seaborn的调色板（palette）功能。通过sns.color_palette()函数设置或查看调色板（palette）,函数返回值是rgb元组的列表。调用sns.palplot()画出palette的每种颜色。sns.palplot(sns.color_palette(palette='summer',n_colors=30))

建立独立配色方案它在任意拥有palette参数的函数内部被使用(在某些需要多种颜色的情况下也可以转入color参数)。

color_palette()可以接受任意的seaborn调色板和matplotlib colormap(除了jet,当然你也不该用这玩意~)。它也可以接收一系列在任意有效的matplotlib格式（RGB tuples, hex color codes, or HTML color names）下设置好的颜色。它的返回值通常是一个RGB元组的list。
最后，无参数调用color_palette()会返回默认的颜色集。
相对应地，set_palette函数可以接受同样的参数，可以为所有图片设置颜色。你依旧可以在with语句下调用color_palette()来暂时修改默认调色板。
[更多颜色控制案例](https://blog.csdn.net/wuwan5296/article/details/78636347)

### 四、分类图

#### 4.1 seaborn.barplot条形图(柱状图)

条形图主要展现的是每个矩形高度的数值变量的中心趋势的估计。

```python
seaborn.barplot(x=None, y=None, hue=None, 
                data=None, order=None, hue_order=None, 
                estimator=<function mean>, ci=95, 
                n_boot=1000, units=None, orient=None, 
                color=None, palette=None, saturation=0.75, 
                errcolor='.26', errwidth=None, capsize=None, 
                dodge=True, ax=None, **kwargs)
```

| 参数            | 类型/含义                  | 作用                                                         |
| --------------- | -------------------------- | ------------------------------------------------------------ |
| x,y,hue         | 数据字段变量名             | -                                                            |
| data            | DataFrame，数组或数组列表  | -                                                            |
| order,hue_order | 字符串列表                 | 显式指定分类顺序，eg. order=[字段变量名1，字段变量名2,...]   |
| estimator       | 可回调函数                 | 设置每个分类箱的统计函数                                     |
| ci              | float或者"sd"或None        | 在估计值附近绘制置信区间的大小，如果是"sd"，则跳过bootstrapping并绘制观察的标准差，如果为None,则不执行bootstrapping,并且不绘制错误条。 |
| n_boot          | int                        | 计算置信区间时使用的引导迭代次数                             |
| orient          | v \| h                     | 图的显示方向(垂直或水平,即横向或纵向)，这通常可以从输入变量的dtype推断得到 |
| color           | matplotlib颜色             | -                                                            |
| palette         | 调试板名称，列表或字典类型 | 设置hue指定的变量的不同级别颜色。                            |
| saturation      | 饱和度：float              |                                                              |
| errcolor        | matplotlib color           | 表示置信区间的线条颜色                                       |
| errwidth        | float                      | 表示误差线的厚度                                             |
| capsize         | float                      | 表示误差线上"帽"的宽度(误差线上的横线的宽度)                 |
| dodge           | bool                       | 使用色调嵌套时，是否应沿分类轴移动元素。                     |

#### 4.2 箱图(Boxplot)

seaborn.boxplot 接口的作用是绘制箱形图以展现与类别相关的数据分布状况。

```python
seaborn.boxplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, width=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5, notch=False, ax=None, **kwargs)
```

箱形图（或盒须图）以一种利于变量之间比较或不同分类变量层次之间比较的方式来展示定量数据的分布。图中矩形框显示数据集的上下四分位数，而矩形框中延伸出的线段（触须）则用于显示其余数据的分布位置，剩下超过上下四分位间距的数据点则被视为“异常值”。

输入数据可以通过多种格式传入，包括：

- 格式为列表，numpy数组或pandas Series对象的数据向量可以直接传递给`x`，`y`和`hue`参数。
- 对于长格式的DataFrame，`x`，`y`，和`hue`参数会决定如何绘制数据。
- 对于宽格式的DataFrame，每一列数值列都会被绘制。
- 一个数组或向量的列表。

| 参数             | 含义/类型                          | 作用                                                         |
| ---------------- | ---------------------------------- | ------------------------------------------------------------ |
| x, y, hue        | `数据`或向量数据中的变量名称，可选 | 用于绘制长格式数据的输入。查看样例以进一步理解。             |
| data             | DataFrame，数组，数组列表，可选    | 用于绘图的数据集。如果`x`和`y`都缺失，那么数据将被视为宽格式。否则数据被视为长格式。 |
| order, hue_order | 字符串列表，可选                   | 控制分类变量（对应的条形图）的绘制顺序，若缺失则从数据中推断分类变量的顺序。 |
| orient           | “v” \| “h”，可选                   | 控制绘图的方向（垂直或水平）。这通常是从输入变量的dtype推断出来的，但是当“分类”变量为数值型或绘制宽格式数据时可用于指定绘图的方向。 |
| color            | matplotlib颜色，可选               | 所有元素的颜色，或渐变调色板的种子颜色。                     |
| palette          | 调色板名称，列表或字典，可选       | 用于`hue`变量的不同级别的颜色。可以从 [`color_palette()`](https://www.cntofu.com/book/172/docs/seaborn.color_palette.html#seaborn.color_palette) 得到一些解释，或者将色调级别映射到matplotlib颜色的字典。 |
| saturation       | float                              | 控制用于绘制颜色的原始饱和度的比例。通常大幅填充在轻微不饱和的颜色下看起来更好，如果您希望绘图颜色与输入颜色规格完美匹配可将其设置为`1`。 |
| width            | float，可选                        | 不使用色调嵌套时完整元素的宽度，或主要分组变量一个级别的所有元素的宽度。 |
| dodge            | bool，可选                         | 使用色调嵌套时，元素是否应沿分类轴移动。                     |
| fliersize        | float，可选                        | 用于表示异常值观察的标记的大小。                             |
| linewidth        | float,可选                         | 构图元素的灰线宽度。                                         |
| whis             | float，可选                        | 控制在超过高低四分位数时IQR的比例，因此需要延长绘制的触须线段。超出此范围的点将被识别为异常值。 |
| notch            | boolean，可选                      | 是否使矩形框“凹陷”以指示中位数的置信区间。还有其他几个参数可以控制凹槽的绘制方式；参见 `plt.boxplot` 以查看关于此问题的更多帮助信息。 |
| ax               | matplotlib轴，可选                 | 绘图时使用的Axes轴对象，否则使用当前Axes轴对象。             |
| kwargs           | 键，值映射                         | 其他在绘图时传给 `plt.boxplot` 的参数。                      |

|        |                    |                                    |
| ------ | ------------------ | ---------------------------------- |
| 返回值 | `ax`：matplotlib轴 | 返回Axes对轴象，并在其上绘制绘图。 |

[中文文档](https://www.cntofu.com/book/172/docs/16.md)

[更多代码参考](https://blog.csdn.net/LuohenYJ/article/details/90677918)

#### 4.3 小提琴图

seaborn.violinplot[中文文档](https://www.cntofu.com/book/172/docs/17.md)

输入数据可以通过多种格式传入，包括：

- 格式为列表，numpy数组或pandas Series对象的数据向量可以直接传递给`x`，`y`和`hue`参数。
- 对于长格式的DataFrame，`x`，`y`，和`hue`参数会决定如何绘制数据。
- 对于宽格式的DataFrame，每一列数值列都会被绘制。
- 一个数组或向量的列表。

```python
seaborn.violinplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, bw='scott', cut=2, scale='area', scale_hue=True, gridsize=100, width=0.8, inner='box', split=False, dodge=True, orient=None, linewidth=None, color=None, palette=None, saturation=0.75, ax=None, **kwargs)
```

```markdown
参数：
x, y, hue：
	数据或向量数据中的变量名称，可选；用于绘制长格式数据的输入。查看样例以进一步理解。
data：DataFrame，数组，数组列表，可选；
	用于绘图的数据集。如果x和y都缺失，那么数据将被视为宽格式。否则数据被视为长格式。
order, hue_order：字符串列表，可选；
	控制分类变量（对应的条形图）的绘制顺序，若缺失则从数据中推断分类变量的顺序。
bw：{‘scott’, ‘silverman’, float}，可选；
	内置变量值或浮点数的比例因子都用来计算核密度的带宽。实际的核大小由比例因子乘以每个分箱内数据的标准差确定。
cut：float，可选；以带宽大小为单位的距离，以控制小提琴图外壳延伸超过内部极端数据点的密度。
	设置为0以将小提琴图范围限制在观察数据的范围内。（例如，在 ggplot 中具有与 trim=True 相同的效果）
scale：{“area”, “count”, “width”}，可选；
	该方法用于缩放每张小提琴图的宽度。若为 area ，每张小提琴图具有相同的面积。若为 count ，小提琴的宽度会根据分箱中观察点的数量进行缩放。若为 width ，每张小提琴图具有相同的宽度。
scale_hue：bool，可选；
	当使用色调参数 hue 变量绘制嵌套小提琴图时，该参数决定缩放比例是在主要分组变量（scale_hue=True）的每个级别内还是在图上的所有小提琴图（scale_hue=False）内计算出来的。
gridsize：int，可选；
	用于计算核密度估计的离散网格中的数据点数目。
width：float，可选；
	不使用色调嵌套时的完整元素的宽度，或主要分组变量的一个级别的所有元素的宽度。
inner：{“box”, “quartile”, “point”, “stick”, None}，可选；
	控制小提琴图内部数据点的表示。若为box，则绘制一个微型箱型图。若为quartiles，则显示四分位数线。若为point或stick，则显示具体数据点或数据线。使用None则绘制不加修饰的小提琴图。
split：bool，可选；
	当使用带有两种颜色的变量时，将split设置为True则会为每种颜色绘制对应半边小提琴。从而可以更容易直接的比较分布。
dodge：bool，可选；
	使用色调嵌套时，元素是否应沿分类轴移动。
orient：“v” | “h”，可选；
	控制绘图的方向（垂直或水平）。这通常是从输入变量的dtype推断出来的，但是当“分类”变量为数值型或绘制宽格式数据时可用于指定绘图的方向。
linewidth：float，可选；
	构图元素的灰线宽度。
color：matplotlib颜色，可选；
	所有元素的颜色，或渐变调色板的种子颜色。
palette：调色板名称，列表或字典，可选；
	用于hue变量的不同级别的颜色。可以从 color_palette() 得到一些解释，或者将色调级别映射到matplotlib颜色的字典。
saturation：float，可选；
	控制用于绘制颜色的原始饱和度的比例。通常大幅填充在轻微不饱和的颜色下看起来更好，如果您希望绘图颜色与输入颜色规格完美匹配可将其设置为1。
ax：matplotlib轴，可选；
	绘图时使用的Axes轴对象，否则使用当前Axes轴对象。

返回值：ax：matplotlib轴；
	返回Axes对轴象，并在其上绘制绘图。
```



#### 4.4 分类散布图

stripplot[细致参考](https://blog.csdn.net/qq_40195360/article/details/86605860)

```python
seaborn.stripplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, jitter=False, dodge=False, orient=None, color=None, palette=None, size=5, edgecolor='gray', linewidth=0, ax=None, **kwargs)
```

x，y，data：输入数据可以多种格式传递，在大多数情况下，使用Numpy或Python对象是可能的，但是更可取的是pandas对象，因为相关的名称将用于对轴进行注释。此外，还可以对分组变量使用分类类型来控制情节元素的顺序。
order：用order参数进行筛选分类类别，例如：order=[‘sun’,‘sat’]；
jitter：抖动项，表示抖动程度，可以使float，或者True；
dodge：重叠区域是否分开，当使用hue时，将其设置为True，将沿着分类轴将不同色调级别的条带分开。
orient：“v” | “h”，vertical（垂直） 和 horizontal（水平）的意思；



#### 4.5 swarmplot(分布密度散点图)

这个函数类似于stripplot()，但是对点进行了调整(只沿着分类轴)，这样它们就不会重叠。这更好地表示了值的分布，但它不能很好地扩展到大量的观测。

```python
seaborn.swarmplot(x=None, y=None, hue=None, data=None, order=None,
 hue_order=None, dodge=False, orient=None, color=None, palette=None, size=5, 
 edgecolor='gray', linewidth=0, ax=None, **kwargs)
```

[更多参考](ttps://blog.csdn.net/qq_39949963/article/details/80750492)

#### 4.5 分类两面网格图

catplot：分类型数据作坐标轴画图

该函数提供了对几个轴级函数的访问，这些函数使用几种可视化表示形式之一显示一个数字变量和一个或多个分类变量之间的关系。其实说白了就是利用kind参数来画前面Categorical plots（分类图）中的任意8个图形。具体如下：[原文链接](https://blog.csdn.net/qq_40195360/article/details/86605860)

```python
seaborn.catplot(x=None, y=None, hue=None, data=None, row=None, col=None, 
col_wrap=None, estimator=<function mean>, ci=95, n_boot=1000, units=None, 
order=None, hue_order=None, row_order=None, col_order=None, kind='strip',
 height=5, aspect=1, orient=None, color=None, palette=None, legend=True, 
 legend_out=True, sharex=True, sharey=True, margin_titles=False, 
 facet_kws=None, **kwargs)
```

- kind：默认strip（分布散点图），也可以选择“point”, “bar”, “count”,
- col、row：将决定网格的面数的分类变量，可具体制定；
- col_wrap：指定每行展示的子图个数，但是与row不兼容；
- row_order, col_order : 字符串列表，安排行和列，以及推断数据中的对象；
- height，aspect：与图像的大小有关；

sharex，sharey：bool, ‘col’or ‘row’，是否共享想，x，y坐标；

#### 4.6 pd.cut

将数据进行离散化、将连续变量进行分段汇总 [引用链接](https://blog.csdn.net/suixuejie/article/details/82383192?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)

pd.cut(x,bins,right=True,labels=None,retbins=False,precision=3,include_lowest=False)

| x              | 一维数组                                                     |
| -------------- | ------------------------------------------------------------ |
| bins           | 整数--将x划分为多少个等距的区间，序列--将x划分在指定序列中，若不在该序列中，则是Nan |
| right          | 是否包含右端点                                               |
| labels         | 是否用标记来代替返回的bins                                   |
| precision      | 精度                                                         |
| include_lowest | 是否包含左端点                                               |

#### 4.7 增强箱图(boxenplot)

增强箱图又称增强盒形图，可以为大数据集绘制增强的箱图。增强箱图通过绘制更多的分位数来提供数据分布的信息。[引用链接](https://blog.csdn.net/zyb228/article/details/101937089)

```python
seaborn.boxenplot(x=None, y=None, hue=None, 
                  data=None, order=None, hue_order=None,
                  orient=None, color=None, palette=None, 
                  saturation=0.75, width=0.8, dodge=True, 
                  k_depth='proportion', linewidth=None, scale='exponential', 
                  outlier_prop=None, ax=None, **kwargs)
```

```markdown
x,y,hue:数据字段变量名(如上表，date,name,age,sex为数据字段变量名)
	作用：根据实际数据，x,y常用来指定x,y轴的分类名称，hue常用来指定第二次分类的数据类别(用颜色区分)
data: DataFrame,
	数组或数组列表
order,hue_order:字符串列表
	作用：显式指定分类顺序，eg. order=[字段变量名1，字段变量名2,...]
orient:方向：v或者h
	作用：设置图的绘制方向(垂直或水平),如何选择：一般是根据输入变量的数据类型(dtype)推断出来。
color:matplotlib 颜色
palette:调色板名称，list类别或者字典
	作用：用于对数据不同分类进行颜色区别
saturation 饱和度：float
width宽度: float
dodge:bool
	作用：若设置为True则沿着分类轴，将数据分离出来成为不同色调级别的条带，否则，每个级别的点将相互叠加
linewidth:float
	作用：设置构图元素的线宽度
```



### 五、关联图

#### 5.1 关联散布图(scatterplot)

```python
seaborn.scatterplot(x=None, y=None, hue=None,
                    style=None, size=None, data=None, 
                    palette=None, hue_order=None, hue_norm=None,
                    sizes=None, size_order=None, size_norm=None, 
                    markers=True, style_order=None, x_bins=None,
                    y_bins=None, units=None, estimator=None, 
                    ci=95, n_boot=1000, alpha='auto', x_jitter=None,
                    y_jitter=None, legend='brief', ax=None, **kwargs)
```

```markdown
data: DataFrame

可选参数
x,y为数据中变量的名称;
	作用：对将生成具有不同颜色的元素的变量进行分组。可以是分类或数字.
size：数据中的名称
	作用：根据指定的名称(列名)，根据该列中的数据值的大小生成具有不同大小的效果。可以是分类或数字。
style:数据中变量名称(比如：二维数据中的列名)
	作用：对将生成具有不同破折号、或其他标记的变量进行分组。
palette:调试板名称，列表或字典类型
	作用：设置hue指定的变量的不同级别颜色。
hue_order:列表(list)类型
	作用：指定hue变量出现的指定顺序，否则他们是根据数据确定的。
hue_norm:tuple或Normalize对象
sizes:list dict或tuple类型
	作用：设置线宽度，当其为数字时，它也可以是一个元组，指定要使用的最大和最小值，会自动在该范围内对其他值进行规范化。
```

[更多案例参考](https://cloud.tencent.com/developer/article/1506466)



#### 5.2 关联线图(折线图)

数据一定是通过DataFrame中传送的[引用](https://cloud.tencent.com/developer/article/1506467)

```python
seaborn.lineplot(x=None, y=None, hue=None, 
                 size=None, style=None, data=None,
                 palette=None, hue_order=None, hue_norm=None, 
                 sizes=None, size_order=None, size_norm=None, 
                 dashes=True, markers=None, style_order=None, 
                 units=None, estimator='mean', ci=95, n_boot=1000,
                 sort=True, err_style='band', err_kws=None,
                 legend='brief', ax=None, **kwargs)
```

```markdown
data:是DataFrame类型的;

可选：下面均为可选
x,y：数据中变量的名称;
hue:数据中变量名称(比如：二维数据中的列名)
	作用：对将要生成不同颜色的线进行分组，可以是分类或数据。
size:数据中变量名称(比如：二维数据中的列名)
	作用：对将要生成不同宽度的线进行分组，可以是分类或数据。
style:数据中变量名称(比如：二维数据中的列名)
	作用：对将生成具有不同破折号、或其他标记的变量进行分组。
palette:调试板名称，列表或字典类型
	作用：设置hue指定的变量的不同级别颜色。
hue_order:列表(list)类型
	作用：指定hue变量出现的指定顺序，否则他们是根据数据确定的。
hue_norm:tuple或Normalize对象
sizes:list dict或tuple类型
	作用：设置线宽度，当其为数字时，它也可以是一个元组，指定要使用的最大和最小值，会自动在该范围内对其他值进行规范化。
units:对变量识别抽样单位进行分组，使用时，将为每个单元绘制一个单独的行。
estimator:pandas方法的名称或回调函数或者None
	作用：用于在同一x水平上聚合y变量的多个观察值的方法，如果为None，则将绘制所有观察结果。
```

[更多案例参考](https://cloud.tencent.com/developer/article/1506467)

#### 5.3 散布图矩阵

变量关系组图

```python
seaborn.pairplot(data, hue=None, hue_order=None, 
                 palette=None, vars=None, x_vars=None,
                 y_vars=None, kind='scatter', diag_kind='auto', 
                 markers=None, height=2.5, aspect=1,
                 dropna=True,plot_kws=None, diag_kws=None,
                 grid_kws=None, size=None)
```

```markdown
data: DataFrame

hue:变量名称
	作用：用颜色将数据进行第二次分组
hue_order:字符串列表
	作用：指定调色板中颜色变量的顺序
palette:调色板
vars:变量名列表
{x,y}_vars:变量名列表
	作用：指定数据中变量分别用于图的行和列，
kind：{"scatter","reg"}
	作用：指定数据之间的关系eg. kind="reg":指定数据的线性回归
diag_kind:{"auto","hist","kde"}
	作用：指定对角线处子图的类型，默认值取决与是否使用hue。参考案例9和案例11
markers:标记
height:标量
	作用：指定图的大小(图都是正方形的，所以只要指定height就行)
{plot，diag，grid} _kws：dicts字典
	作用：指定关键字参数的字典
```

[更多案例参考](https://cloud.tencent.com/developer/article/1517208)

#### 5.4 分面网格关联图

seaborn.relplot()

```python
seaborn.relplot(x=None, y=None, hue=None, size=None, 
                style=None, data=None, row=None, col=None,
                col_wrap=None, row_order=None, col_order=None, 
                palette=None, hue_order=None, hue_norm=None, 
                sizes=None, size_order=None, size_norm=None, 
                markers=None, dashes=None, style_order=None, 
                legend='brief', kind='scatter', height=5, 
                aspect=1, facet_kws=None, **kwargs)
```

```markdown
必须的参数x,y,data
其他参数均为可选；
x,y：数据中变量的名称;
data:是DataFrame类型的;
可选：下面均为可选
hue:数据中的名称 
	对将生成具有不同颜色的元素的变量进行分组。可以是分类或数字.
row，col：数据中变量的名称
	分类变量将决定网格的分面。
col_wrap：int
	这个变量设置可以将多列包装以多行的形式展现(有时太多列展现，不便利)，但不可以将多行以多列的形式展现。
size：数据中的名称
	根据指定的名称(列名)，根据该列中的数据值的大小生成具有不同大小的效果。可以是分类或数字。
```

[更多案例参考](https://cloud.tencent.com/developer/article/1506462)



### 六、其他图形

#### 6.1 分布图

- 直方图

seaborn.distplot() 直方图，质量估计图，核密度估计图

该API可以绘制分别直方图和核密度估计图，也可以绘制直方图和核密度估计图的合成图 通过设置默认情况下，是绘制合成图，设置情况图下：

hist=True:表示要绘制直方图(默认情况为True)，若为False，则不绘制

kde=True:表示要绘制核密度估计图(默认情况为True),若为False,则绘制

```python
seaborn.distplot(a, bins=None, hist=True, 
                 kde=True, rug=False, fit=None, 
                 hist_kws=None, kde_kws=None, rug_kws=None,
                 fit_kws=None, color=None, vertical=False,
                 norm_hist=False, axlabel=None,
                 label=None, ax=None)
```

```markdown
a: Series, 一维数组或列表
要输入的数据，如果设置name属性，则该名称将用于标记数据轴；
以下是可选参数:
bins: matplotlib hist()的参数 或者 None
	作用：指定直方图规格，若为None，则使用Freedman-Diaconis规则,该规则对数据中的离群值不太敏感，可能更适用于重尾分布的数据。它使用 bin 大小  
[2∗IQR(X(:))∗numel(X)(−1/4),2∗IQR(Y(:))∗numel(Y)(−1/4)][2∗IQR(X(:))∗numel(X)(−1/4),2∗IQR(Y(:))∗numel(Y)(−1/4)] ，其中 IQR 为四分位差。
hist:bool
	是否绘制(标准化)直方图
kde:bool
	是否绘制高斯核密度估计图
rug:bool
	是否在支撑轴上绘制rugplot()图
{hist，kde，rug，fit} _kws：字典
	底层绘图函数的关键字参数
color:matplotlib color
	该颜色可以绘制除了拟合曲线之外的所有内容
vertical:bool
	如果为True,则观察值在y轴上，即水平横向的显示
```

[更多案例参考](https://cloud.tencent.com/developer/article/1512635)

- 核函数密度估计图kdeplot()

主要用来拟合并绘制单变量或双变量核密度估计值。

```python
seaborn.kdeplot(data, data2=None, shade=False, 
                vertical=False, kernel='gau', bw='scott',
                gridsize=100, cut=3, clip=None,
                legend=True, cumulative=False,
                shade_lowest=True, cbar=False, 
                cbar_ax=None, cbar_kws=None, ax=None, **kwargs)
```

```markdown
shade:阴影：bool类型
作用：设置曲线下方是否添加阴影，如果为True则在曲线下方添加阴影
(如果数据为双变量则使用填充的轮廓绘制)，若为False则，不绘制阴影

cbar:bool类型
作用：如果为True则绘制双变量KDE图，并添加颜色条
```

[更多案例参考](https://cloud.tencent.com/developer/article/1512637)

- 核函数密度估计图kdeplot()

主要用来拟合并绘制单变量或双变量核密度估计值。

```python
seaborn.kdeplot(data, data2=None, shade=False, 
                vertical=False, kernel='gau', bw='scott',
                gridsize=100, cut=3, clip=None,
                legend=True, cumulative=False,
                shade_lowest=True, cbar=False, 
                cbar_ax=None, cbar_kws=None, ax=None, **kwargs)
```

```markdown
shade:阴影：bool类型
作用：设置曲线下方是否添加阴影，如果为True则在曲线下方添加阴影
(如果数据为双变量则使用填充的轮廓绘制)，若为False则，不绘制阴影

cbar:bool类型
作用：如果为True则绘制双变量KDE图，并添加颜色条
```

[更多案例参考](https://cloud.tencent.com/developer/article/1512637)



#### 6.2 双变量关系图jointplot()

在默认情况下双变量关系图是散点图与直方图组合的联合直方图，可以通过设置kind来改变联合直方图。

```python
seaborn.jointplot(x, y, data=None, kind='scatter', 
                  stat_func=None, color=None, height=6,
                  ratio=5, space=0.2, dropna=True, 
                  xlim=None, ylim=None, joint_kws=None,
                  marginal_kws=None, annot_kws=None, **kwargs)
```

```markdown
x,y,hue:数据字段变量名(如上表，date,name,age,sex为数据字段变量名)
data: DataFrame
kind：{"scatter"| "reg"| "resid"| "kde"| "hex"}
	作用：指定要绘制的类型
color : matplotlib color
height : 数字
	作用：指定图的大小(图是正方形的)
ratio:数字
	作用：指定主轴(x,y轴)与边缘轴(正方形四边除x,y轴外的其它轴)高度的比率
space：数字
	作用：指定主轴与边缘轴之间的空间
dropna : bool
	作用：如果为True,则删除x和y中缺少的观测值
```

[更多案例参考](https://cloud.tencent.com/developer/article/1517207)

#### 6.3 热力图heatmap()

将矩形数据绘制成颜色编码矩阵

```python
seaborn.heatmap(data, vmin=None, vmax=None, 
                cmap=None, center=None, robust=False,
                annot=None, fmt='.2g', annot_kws=None,
                linewidths=0, linecolor='white', cbar=True, 
                cbar_kws=None, cbar_ax=None, square=False,
                xticklabels='auto', yticklabels='auto', 
                mask=None, ax=None, **kwargs)
```

```markdown
data:矩形数据集
	可以强制转换为ndarray的2D数据，如果提供了Pandas DataFrame,则索引/列信息将用于标记列和行。
vmin,vmax：float
	作用：锚定颜色图的值
cmap:matplotlib颜色图名称或对象，或者颜色列表
	作用：指定从数据值到颜色空间的映射。如果未提供，则默认值取决于是否设置了中心。
center:float
	作用：绘制不同数据时将颜色图居中的值，如果未指定，则使用此参数将更改默认的cmap
robust:bool
	作用：如果不为True且vmin或vmax不存在，则使用稳健的分位数而不是极值来计算色图范围。
linewidths:线宽 float
	作用：将划分每个单元格的线宽度
linecolor:线颜色
	作用：指定每个单元格的线的颜色
cbar:bool
	作用：指定是否绘制颜色条
```

[更多案例参考](https://cloud.tencent.com/developer/article/1517211)

#### 6.4 线性回归图regplot()

利用线性回归模型对数据进行拟合。

```python
seaborn.regplot(x, y, data=None,x\_estimator=None, 
                x\_bins=None,x\_ci='ci', scatter=True,
                fit\_reg=True, ci=95, n\_boot=1000,
                units=None, order=1, logistic=False,
                lowess=False, robust=False, logx=False,
                x\_partial=None, y\_partial=None,
                truncate=False, dropna=True,
               x\_jitter=None, y\_jitter=None, label=None,
                color=None, marker='o', scatter\_kws=None,
                line\_kws=None, ax=None)
```

[更多案例参考](https://cloud.tencent.com/developer/article/1517210)

#### 6.5 分面网格图

通过FacetGrid绘制网格，在网格中添加子图：

第一步使用`sns.FacetGrid`构造函数`sns.FacetGrid(data,row=None,col=None,hue=None,col_wrap=None)`

第二部使用`FacetGrid.map`方法:`FacetGrid.map(func,*args,**kwargs)`

[更多说明1](https://blog.csdn.net/weixin_42398658/article/details/82960379)

#### 6.7 散点图scatterplot()

可以通过调整颜色、大小和样式等参数来显示数据之间的关系。

```python
seaborn.scatterplot(x=None, y=None, hue=None,
                    style=None, size=None, data=None, 
                    palette=None, hue_order=None, hue_norm=None,
                    sizes=None, size_order=None, size_norm=None, 
                    markers=True, style_order=None, x_bins=None,
                    y_bins=None, units=None, estimator=None, 
                    ci=95, n_boot=1000, alpha='auto', x_jitter=None,
                    y_jitter=None, legend='brief', ax=None, **kwargs)
```

```markdown
data: DataFrame
可选参数
x,y为数据中变量的名称;
	作用：对将生成具有不同颜色的元素的变量进行分组。可以是分类或数字.
size：数据中的名称
	作用：根据指定的名称(列名)，根据该列中的数据值的大小生成具有不同大小的效果。可以是分类或数字。
style:数据中变量名称(比如：二维数据中的列名)
	作用：对将生成具有不同破折号、或其他标记的变量进行分组。
palette:调试板名称，列表或字典类型
	作用：设置hue指定的变量的不同级别颜色。
hue_order:列表(list)类型
	作用：指定hue变量出现的指定顺序，否则他们是根据数据确定的。	
hue_norm:tuple或Normalize对象
sizes:list dict或tuple类型
	作用：设置线宽度，当其为数字时，它也可以是一个元组，指定要使用的最大和最小值，会自动在该范围内对其他值进行规范化。
```

[更多案例参考](https://cloud.tencent.com/developer/article/1506466)




