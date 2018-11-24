## Chinese Article Segment 2018-11

#### 1.preprocess

clean() 删去无用字符，prepare() 打乱后划分训练数据、测试标签和数据

#### 2.explore

统计词汇、长度、bigram 的频率，条形图可视化

#### 3.build

word2vec() 训练词向量，通过 bigram 得到条件频率、概率分布

#### 4.segment

predict() 先分别进行前向、后向最大匹配，结果相同则直接返回

不同则计算每句的平均对数概率、未录 bigram 可使用 divide、neural 平滑

#### 5.eval

get_cut_ind() 得到 pred、label 的切分位，计算查准率、查全率、f1 值