import tensorflow  as tf 
import numpy as np
import os

class RNN(object):
	def __init__(self,features,batch_size = 64, max_time = 50, num_units = 128, layers = 2, learning_rate = 0.001,
		grad_clip = 5, sample = False, train_keep_prob = 0.5, use_embedding = False, embedding_size = 128):
	if sample is True: 
		batch_size = 1; max_time = 1

	self.features = features # 特征维度
	self.batch_size = batch_size # 批数量
	self.max_time = max_time # 序列长度
	self.num_units = num_units # 隐藏神经元参数个数
	self.layers = layers # cell层数
	self.learning_rate = learning_rate # 学习率
	self.grad_clip = grad_clip # 梯度范围
	self.train_keep_prob = train_keep_prob # 保持率
	self.use_embedding = use_embedding # 是否使用嵌入
	self.embedding_size = embedding_size # 嵌入映射大小

	tf.reset_default_graph() # 重置图
    
    """
    输入层 
    1.原始输入
    2.label输入
    3.保持率
    4.lstm的输入
    """
	def build_inputs(self):
		with tf.name_scope('inputs'):
			self.inputs = tf.placeholder(tf.int32, shape=(self.batch_size,self.max_time), name='inputs')
			self.targets = tf.placeholder(tf.int32, shape=(self.batch_size,self.max_time), name='targets')
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
			if self.use_embedding is False:
				self.lstm_inputs = tf.one_hot(self.inputs,self.features) # 把inputs按features展开
			else:
				embedding = tf.get_variable('embedding',[self.features,self.embedding_size])
				self.lstm_inputs = tf.nn.embedding_lookup(embedding, self.inputs)
	"""
	lstm层
	"""
	def build_lstm(self):
		# 创建单个lstm格
		def create_single_cell(num_units,keep_prob):
			lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units) #构建基本格
			return tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=keep_prob) #控制输出的比率dropout

		with tf.name_scope('lstm'):
			cells = tf.nn.rnn_cell.MultiRNNCell([create_single_cell(self.num_units,self.keep_prob) for _ in range(self.layers)])
			self.initial_state = cells.zero_state(self.batch_size,tf.float32)
			# 通过lstm_outputs得到概率 (tf.concat)
			"""
    		lstm_output: 包含RNN小区在每个时刻的输出。
			final_state包含处理完所有输入后的RNN状态。
			请注意，与lstm_output不同，此信息不包含有关每个时间步的信息，而只包含最后一个信息（即最后一个之后的状态）。
    		"""
			#在时间维度上展开cell，即建立循环网络 lstem_output.shape = [batch_size * num_units]
			self.lstm_outputs, self.final_state = tf.nn.dynamic_rnn(cells, self.lstm_inputs,initial_state=self.initial_state)

			x = tf.reshape(tf.concat(self.lstm_outputs,1),[-1,self.num_units])
			with tf.variable_scope('softmax'):
				W = tf.Variable(tf.truncated_normal([self.num_units,self.features], stddev=0.1))
				b = tf.Variable(tf.zeros(self.features))

			self.logits = tf.matmul(x,W) + b # 结果shape为 
			self.prediction = tf.nn.softmax(self.logits,name='prediction')
	"""
	loss层
	"""
	def build_loss(self):
		with tf.name_scope('loss'):
			y_one_hot = tf.one_hot(self.targets, self.features) #把targets按features展开
			y_label = tf.reshape(y_one_hot, self.logits.get_shape()) #把label的shape变成和logits一致
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_label))
