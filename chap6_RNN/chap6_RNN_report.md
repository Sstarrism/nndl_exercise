# Chap6_RNN 作业报告

## 1. 任务执行概述
- 代码位置：`chap6_RNN/tangshi_for_pytorch/main.py`, `chap6_RNN/tangshi_for_pytorch/rnn.py`
- 已完成：PyTorch 版本的补全
  - `rnn.py` 中 `self.rnn_lstm` 的定义
  - `rnn.py` 中 `forward` 的 LSTM 前向传播（初始隐状态和细胞状态初始化）
  - `main.py` 中 `import rnn as rnn_lstm`
  - 修复路径问题：`poems.txt` 路径改为 `./chap6_RNN/tangshi_for_pytorch/poems.txt`
- 补全其他文件：
  - `Learn2Carry-exercise.ipynb`：补全 `myRNNModel.call` 方法，实现加法进位RNN模型
  - `poem_generation_with_RNN-exercise.ipynb`：补全 `myRNNModel.call` 和 `train_one_step` 方法，实现诗歌生成RNN模型
- 生成诗歌起始词：`日`、`红`、`山`、`夜`、`湖`、`海`、`月`（已调用 `gen_poem` 方法）

## 2. RNN / LSTM / GRU 模型解释
### 2.1 RNN（Recurrent Neural Network）
- 基本概念：序列数据逐步输入，隐层状态与当前输入一起进入下一时间步。
- 公式（单层简单RNN）：
  - $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$
  - $y_t = W_{hy} h_t + b_y$
- 优点：建模时间序列依赖；结构简单。
- 缺点：梯度消失/爆炸，长序列记忆困难。

### 2.2 LSTM（Long Short-Term Memory）
- 专为解决长依赖设计，引入门控机制：输入门、遗忘门、输出门、细胞状态。
- 公式：
  - $f_t=\sigma(W_f[x_t,h_{t-1}]+b_f)$
  - $i_t=\sigma(W_i[x_t,h_{t-1}]+b_i)$
  - $o_t=\sigma(W_o[x_t,h_{t-1}]+b_o)$
  - $\tilde{C}_t=\tanh(W_C[x_t,h_{t-1}]+b_C)$
  - $C_t=f_t * C_{t-1}+i_t * \tilde{C}_t$
  - $h_t=o_t * \tanh(C_t)$
- 优点：能跨时间步存储和过滤信息，缓解梯度消失。

### 2.3 GRU（Gated Recurrent Unit）
- 门控简化版本：更新门和重置门，合并细胞状态与隐藏状态。
- 公式：
  - $z_t=\sigma(W_z[x_t,h_{t-1}]+b_z)$
  - $r_t=\sigma(W_r[x_t,h_{t-1}]+b_r)$
  - $\tilde{h}_t=\tanh(W_h[x_t, r_t*h_{t-1}]+b_h)$
  - $h_t=(1-z_t)*h_{t-1}+z_t*\tilde{h}_t$
- 优点：结构更简单、参数更少、性能与LSTM接近。

## 3. 诗歌生成过程
1. 数据预处理
   - 文件：`poems.txt`
   - 每行 `标题:内容`（reader 可能是 `tangshi.txt` 或 `poems.txt`）
   - 添加开始符 `G`，结束符 `E`，过滤不合条件的句子（符号、长度、重复字符）。
   - 统计词频，构建 `word -> index` 字典，并将诗转换为索引序列。
   - `generate_batch` 输出 `(x_batches, y_batches)`，序列标签为后移一位。

2. 模型定义（`rnn.py`）
   - 词嵌入层：`nn.Embedding`（随机初始化）
   - LSTM：`nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_dim, num_layers=2, batch_first=True)`
   - 线性输出：`nn.Linear(lstm_hidden_dim, vocab_len)`
   - LogSoftmax：`nn.LogSoftmax(dim=1)`（与 `NLLLoss` 协同使用）

3. 训练流程（`main.py` 的 `run_training()`）
   - 每轮遍历 batch（`BATCH_SIZE=100`）
   - 对于每一个样本，转换成 LongTensor, 前向输出并累计 `NLLLoss`。
   - 平均损失后反向，梯度裁剪 `clip_grad_norm`，优化器 `RMSprop(lr=1e-2)`。
   - 每 20 个 batch 保存一次 `poem_generator_rnn`。

4. 诗歌生成（`gen_poem(begin_word)`）
   - 载入模型权重：`rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))`
   - 句子初始为 `begin_word`，循环预测单词：
     - 把当前生成句子转索引，输入模型，取最后一步输出概率向量。
     - 选择概率最高词，拼接到句子上。
     - 直至出现 `end_token` 或长度超过 30。

5. 输出格式化
   - `pretty_print_poem` 按 `。` 分句并显示全文。

## 4. 起始词生成实验结果
- 生成词：`日、红、山、夜、湖、海、月、君`（8 个词，含题意 7 词）
- 代码位置：`main.py` 最后调用：
  - `pretty_print_poem(gen_poem("日"))`
  - ...
  - `pretty_print_poem(gen_poem("君"))`
- 结果示例（示例输出因随机/模型未训练时间不同会变）：
  - `日...` 
  - `红...`
  - `山...`
  - `夜...`
  - `湖...`
  - `海...`
  - `月...`
  - `君...`

> 注：此环境中未安装 `torch`，无法直接训练得出最后文本。实际报告中应插入作业截图：
> - 训练过程日志截图（Pytorch 版本要求）
> - 各个起始词生成结果截图

## 5. 补全代码位置说明
- `rnn.py`:
  - `RNN_model.__init__`：构建 `self.rnn_lstm`
  - `RNN_model.forward`：实现 LSTM 前向数据流，初始化 `h0/c0` 为零
- `main.py`：
  - `import rnn as rnn_lstm`（修复模块变量一致性）
  - 修复路径：`poems.txt` 改为 `./chap6_RNN/tangshi_for_pytorch/poems.txt`
- `Learn2Carry-exercise.ipynb`：
  - `myRNNModel.call`：实现嵌入、拼接、RNN、Dense 输出
- `poem_generation_with_RNN-exercise.ipynb`：
  - `myRNNModel.call`：实现嵌入、RNN、Dense 输出
  - `train_one_step`：实现梯度下降训练步骤

## 6. 实验总结
- 本作业通过字符级语言模型完成古诗生成，训练数据后移生成下一字。
- LSTM 的门控结构适合诗歌这种长依赖文本。
- 核心是数据准备、词表统一、`nll` loss 和 `logsoftmax` 的搭配。
- 实验要求启发词起点的生成机制已实现，评估指标可用“可读性”“押韵”等主观评价。

---

> 后续可继续优化：
> 1) 使用 `temperature` 软化采样而非贪心；
> 2) 增加 `dropout` 或 `梯度裁剪` 恰当值；
> 3) 增加“古诗格律规则”判据（5言/7言、平仄、对仗）。
