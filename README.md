## Chinese Article Segment 2018-11

#### 1.preprocess

clean() 删去无用字符，打乱后 train 70% / dev 20% / test 10% 划分

#### 2.represent

convert() 分别删去、记录空格得到 sent、label，pad() 填充为相同长度

#### 3.build

通过单向、双向 rnn 构建序列标注模型，根据 dev_loss、trap_count 控制训练

#### 4.segment

predict() 通过原句长度得到 mask_pred，在 prob > thre 的字后插入空格