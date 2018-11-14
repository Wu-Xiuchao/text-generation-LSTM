import tensorflow as tf 
from text import Text 
from model import RNN 
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('num_units',128,'隐藏神经元个数')
tf.flags.DEFINE_integer('layers',2,'lstm网络层数') 
tf.flags.DEFINE_boolean('use_embedding', False, '是否embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'embedding大小')
tf.flags.DEFINE_string('converter_path', '', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', '', 'checkpoint path')
tf.flags.DEFINE_integer('max_length', 30, '生成文本的长度')

def main():
	converter = Text(filename=FLAGS.converter_path)
	if os.path.isdir(FLAGS.checkpoint_path):
		FLAGS.checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)

	model = RNN(converter.vocab_size,sample=True,
		num_units = FLAGS.num_units,
		layers = FLAGS.layers,
		use_embedding = FLAGS.use_embedding,
		embedding_size = FLAGS.embedding_size)
	#建好图结构后导入参数就可以使用网络了
	model.load(FLAGS.checkpoint_path)

	arr = model.sample(FLAGS.max_length,converter.vocab_size)
	print(converter.arr_to_text(arr))

if __name__ == '__main__':
	tf.app.run()
