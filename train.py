from model import RNN
import function
import os
import tensorflow as tf 
from text import Text 

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name','default','模型的名称')
tf.flags.DEFINE_integer('batch_size',100,'序列的数量')
tf.flags.DEFINE_integer('max_time',100,'序列时长')
tf.flags.DEFINE_integer('num_units',128,'隐藏神经元个数')
tf.flags.DEFINE_integer('layers',2,'lstm网络层数')
tf.flags.DEFINE_boolean('use_embedding',False,'是否Embedding')
tf.flags.DEFINE_integer('embedding_size',128,'Embedding大小')
tf.flags.DEFINE_float('learning_rate',0.01,'学习率')
tf.flags.DEFINE_float('train_keep_prob',0.5,'Dropout')
tf.flags.DEFINE_string('input_file','','训练的文件')
tf.flags.DEFINE_integer('max_steps',10000,'迭代次数')
tf.flags.DEFINE_integer('max_vocab',3500,'字典容量')

def main(_):
	model_path = os.path.join('model',FLAGS.name) #建立模型地址
	if os.path.exists(model_path) is False:
		os.makedirs(model_path) #若模型地址不存在，递归建立文件
	with open(FLAGS.input_file,'r',encoding='utf-8') as file:
		text = file.read() 
	converter = Text(text,FLAGS.max_vocab)
	converter.save_to_file(os.path.join(model_path,'converter.pkl')) #把文本保存起来

	arr = converter.text_to_arr(text)#把文本转为数组
	g = function.batch_generator(arr,FLAGS.batch_size,FLAGS.max_time) #生成器
	model = RNN(converter.vocab_size,
		        batch_size = FLAGS.batch_size,
		        max_time = FLAGS.max_time,
		        num_units = FLAGS.num_units,
		        layers = FLAGS.layers,
		        learning_rate = FLAGS.learning_rate,
		        train_keep_prob = FLAGS.train_keep_prob,
		        use_embedding = FLAGS.use_embedding,
		        embedding_size = FLAGS.embedding_size)
	model.train(g,FLAGS.max_steps,model_path) #训练 

if __name__ == '__main__':
	tf.app.run()
