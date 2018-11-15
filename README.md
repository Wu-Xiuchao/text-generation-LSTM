# text-generation-LSTM (Updating...)
--------------------------
![LSTM示意图](https://github.com/Wu-Xiuchao/text-generation-LSTM/blob/master/picture/lstm.png)
LSTM网络主要有遗忘门，更新门和输出门三个部分.  
参考了http://colah.github.io/posts/2015-08-Understanding-LSTMs/  
整个代码部分学习的是https://github.com/hzy46/Char-RNN-TensorFlow 所给的示例代码   
我对整个代码做了详细的注释，并希望在未来改进一些  
对于各部分的shape变换做了整理，整个网络训练架构以及shape变换如下所示：  
![](https://github.com/Wu-Xiuchao/text-generation-LSTM/blob/master/picture/shape%E5%8F%98%E6%8D%A2.png)  
总共5个python文件  
text.py 主要是对文本处理的一个文件  
model.py 是LSTM模型文件  
function.py 是各种用到的方法文件  
train.py 是训练文件  
sample.py 是生成文本文件
