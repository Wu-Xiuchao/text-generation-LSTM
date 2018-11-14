import tensorflow  as tf 
import numpy as np
import os
import function

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
	self.build_inputs()
	self.build_lstm()
	self.build_loss()
	self.build_opt()
	self.saver = tf.train.Saver()
    
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

			self.logits = tf.matmul(x,W) + b # 结果shape为 [batch_size*max_time, num_units]
			self.prediction = tf.nn.softmax(self.logits,name='prediction')
	"""
	loss层
	"""
	def build_loss(self):
		with tf.name_scope('loss'):
			y_one_hot = tf.one_hot(self.targets, self.features) #把targets按features展开
			y_label = tf.reshape(y_one_hot, self.logits.get_shape()) #把label的shape变成和logits一致
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_label))

	"""
	opt层
	"""
	def build_opt(self):
		"""
		１．在solver中先设置一个grad_clip
		２．在前向传播与反向传播之后，我们会得到每个权重的梯度diff，这时不像通常那样直接使用这些梯度进行权重更新，
		而是先求所有权重梯度的平方和sumsq_diff，如果sumsq_diff > grad_clip，则求缩放因子scale_factor = clip_gradient / sumsq_diff。
		这个scale_factor在(0,1)之间。如果权重梯度的平方和sumsq_diff越大，那缩放因子将越小。 
		３．最后将所有的权重梯度乘以这个缩放因子，这时得到的梯度才是最后的梯度信息。
		这样就保证了在一次迭代更新中，所有权重的梯度的平方和在一个设定范围以内，这个范围就是grad_clip。
		出自博客:https://blog.csdn.net/u013713117/article/details/56281715
		"""
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss,tvars),self.grad_clip)
		train_op = tf.train.AdamOptimizer(self.learning_rate)
		self.optimizer =  train_op.apply_gradients(zip(grads,tvars))

	"""
	训练
	batch_generator 生成器
	max_steps 迭代次数
	save_path 模型保存路径
	"""
	def train(self, batch_generator, max_steps, save_path):
		self.session = tf.Session()
		with self.session as sess:
			sess.run(tf.global_variable_initializer()) #初始化全局变量
			new_state = sess.run(self.initial_state)
			step = 0
			for x,y in batch_generator:
				step += 1
				feed = {self.inputs:x,self.targets:y,self.keep_prob:self.train_keep_prob,self.initial_state:new_state}
				batch_loss, new_state, _ = sess.run([self.loss,self.final_state,self.optimizer],feed_dict=feed)
				if step % 100 == 0:
					print('step: {}/{}------------>loss: {:.4f}'.format(step,max_steps,batch_loss))
				if step >= max_steps:
					break
			self.saver.save(sess,os.path.join(save_path,'model'),global_step=step)
	"""
	生成文字
	n_samples 是生成的字数
	vocab_size 是字典总字数
	"""
	def sample(self, n_samples, vocab_size):
		samples = [] # 建立一个空的数组
		sess = self.session
		new_state = sess.run(self.initial_state) #初始化初始状态
		preds = np.ones((vocab_size,)) #建立长度为vocab_size的一维向量
		c = function.pick_top_n(preds, vocab_size)
		for i in range(n_samples):
			x = np.zeros((1,1))
			x[0,0] = c
			feed = {self.inputs:x,self.keep_prob:1.,self.initial_state:new_state}
			preds, new_state = sess.run([self.prediction,self.final_state],feed_dict=feed)

			c = function.pick_top_n(preds, vocab_size)
			samples.append(c)
		return np.array(samples)

	"""
	加载模型
	"""
	def load(self, checkpoint):
		self.session = tf.Session()
		self.saver.restore(self.session,checkpoint)
