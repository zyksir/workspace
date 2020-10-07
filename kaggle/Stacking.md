[TOC]

古人云，三个臭皮匠，顶个诸葛亮，在机器学习中也有类似的情况。Ensemble 就可以让几个**独立**的弱模型胜过一个强模型。一般常见的模型融合方法有 Voting、Bagging、Boosting 和 Stacking。Bagging 和 Boosting 感觉同一种模型通过修改训练数据来达到增强的效果，感兴趣的小伙伴可以看看[Understanding Random Forests](https://github.com/glouppe/phd-thesis)和[Adaboosting](https://towardsdatascience.com/understanding-adaboost-2f94f22d5bfe)。本文就将介绍 Voting 和 Stacking 这两种模型融合的方法并给出参考代码，它们是可以结合多种模型，达到一个更优的解。

#### Voting & Averaging

##### How it works

- 对于分类问题，我们采用投票机制；对于回归问题，我们可以使用求平均值的方法。

- 通常的来说，给表现较好的模型加一个权重可以让模型的效果有更好的提升。

- 代码可以参考[这里](https://github.com/MLWave/Kaggle-Ensemble-Guide/tree/master/src)。

- 取平均值还可以考虑几何平均数和排名平均法，一般的来说其效果会比代数平均数更好。
  $$
  \left(\prod_{i=1}^{n} x_{i}\right)^{\frac{1}{n}}=\sqrt[n]{x_{1} x_{2} \cdots x_{n}}  \\
  newRank_i = \frac{\sum_{j=1}^{n}rank^j_i}{n}
  $$
  

##### Why it works

**voting**

- 以一个二分类问题为例，对于3个**独立**的准确率都为0.7的分类器，如果采用投票机制，准确率就会上升为$ 0.7 \times 0.7 \times 0.7 + 3 \times 0.7 \times 0.7 \times 0.3 = 0.7838  $。

**Averaging**

- Averaging可以减少过拟合，看下图作为例子：
  ![Learning from noise](/Users/yikaizhu/github/workspace/kaggle/averaging.png)

  



#### Stacking & Blending

##### How it works

- Stacking来自于[Wolpert 1992 paper](https://www.researchgate.net/publication/222467943_Stacked_Generalization)(该作者还提出了`天底下没有免费的午餐`理论，不得不感慨大佬就是 nb。)， Blending 是 Netflix 比赛中几个大牛对Stacking 的改进。

![img](/Users/yikaizhu/github/workspace/kaggle/stacking.png)

- Stacking伪代码如下，可以参考着上面的图理解。(下面代码对 Test Data 的处理方式和上图有所不同，上图是取平均，下面代码是用所有数据训一个新的模型然后去预测。)：
  1. 将数据分成 k 份(KFold)
  2. 对其中的每一份，用剩下k-1份作为训练集训练 first-stage 模型，然后给这一份数据打分；
     k 遍之后，我们就得到了整个数据集在一个新的维度上的特征
  3. 对 n 个 first-stage 模型重复步骤2，我们就得到了关于原先数据集的一个 n 维的特征
  4. 基于这个 n 维的特征，训练second-stage模型。



- Blending 修改了 KFold 这个 part。它选择将整个数据分成一大一小两份(e.g 9 : 1)，小份专门用来训练first-stage模型，然后给大数据大份；大份专门用来训练 second-stage 模型。这个办法感觉不太适用于 BERT 这种模型，本身训练数据就少，而且看了很多kaggle 提交，用 Stacking 的更多一点。

##### why it works

第一层的模型提供了看待问题的各式各样的角度，第二层模型可以获得很多高质量、多角度的信息，因此效果可以得到提升。

> It is usually desirable that the level 0 generalizers are of all “types”, and not just simple variations of one another (e.g., we want surface-fitters, Turing-machine builders, statistical extrapolators, etc., etc.). In this way all possible ways of examining the learning set and trying to extrapolate from it are being exploited. This is part of what is meant by saying that the level 0 generalizers should “span the space”.
>
> […] stacked generalization is a means of non-linearly combining generalizers to make a new generalizer, to try to optimally integrate what each of the original generalizers has to say about the learning set. The more each generalizer has to say (which isn’t duplicated in what the other generalizer’s have to say), the better the resultant stacked generalization.
> Wolpert (1992) Stacked Generalization



附上Stacking 的代码，这里还有一份更加详细的参考代码[Kaggle Titanic](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)：

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier

# 使用 StratifiedKFold 保证了每次 Split 中各个类别的比例都是相似的
def get_out_of_fold_predictions(X, y, models):
    meta_X, meta_y = list(), list()
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    for train_ix, test_ix in kfold.split(X, y):
        fold_yhats = list()
        train_X, test_X, train_y, test_y = X[train_ix], X[test_ix], y[train_ix], y[test_ix]
        meta_y.extend(test_y)
        for model in models:
            model.fit(train_X, train_y)
            yhat = model.predict_proba(test_X)
            fold_yhats.append(yhat)
        meta_X.append(np.hstack(fold_yhats))
    return np.vstack(meta_X), np.asarray(meta_y) # meta_X : (N, n), meta_Y : (N, )

def fit_base_models(X, y, models):
    for model in models:
        model.fit(X, y)

# 这里我们也可以用其他模型
def fit_meta_model(X, y):
    model = ExtraTreesClassifier()
    model.fit(X, y)
    return model

def make_predictions(X, models, meta_model):
    meta_X = list()
    for model in models:
        yhat = model.predict_proba(X)
        meta_X.append(yhat)
    meta_X = np.hstack(meta_X)
    return meta_model.predict(meta_X)

'''
models = get_models()
meta_X, meta_y = get_out_of_fold_predictions(train_X, train_y, models)
fit_base_model(train_X, train_y, models)
meta_model = fit_meta_model(meta_X, meta_y)
make_predictions(test_X, models, meta_model)
'''
```



图来自 ：[Kaggle机器学习之模型融合（stacking）心得](https://zhuanlan.zhihu.com/p/26890738)

代码参考自：[EDA to Ensembles](https://www.kaggle.com/doomdiskday/full-tutoria-eda-to-ensembles-embeddings-zoo)

主要参考的博客[KAGGLE ENSEMBLING GUIDE](https://mlwave.com/kaggle-ensembling-guide/) 这是一篇非常棒的博客，后面 Feature weighted linear stacking, Stacking unsupervised learned features, 和Online Stacking还没看，真的难 orz，留着以后整理

