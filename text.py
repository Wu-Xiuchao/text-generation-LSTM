import numpy as np 
import pickle

"""
这是一个文本类，只需传入文本text,当然也可以指定max_vocab
功能包括：
1、获取字典长度
2、把文本转为数组
3、把数组转为文本
4、保存字典
"""
class Text(object):
	def __init__(self, text=None, max_vocab=5000,filename=None):
		if filename is not None:
			with open(filename,'rb') as file:
				self.vocab = pickle.load(file)
		else:
			vocab = set(text)
			"""
			取出现次数高的前max_vocab个word
			"""
			vocab_count = {word:text.count(word) for word in vocab}
			vocab_count = sorted(vocab_count.items(),key=lambda x:x[1], reverse=True)
			vocab_count = vocab_count[:max_vocab] # 在这里保证字数不超过max_vocab
			self.vocab = [x[0] for x in vocab_count] # 通过下标可以查找对应的字
		self.word_to_index = {c:i for i,c in enumerate(self.vocab)} # 通过字查找对应的下标

	@property
	def vocab_size(self):
		return len(self.vocab) + 1

	# 把文本转为数组
	def text_to_arr(self,text):
		return np.array([self.word_to_index[word] if word in self.word_to_index else len(self.vocab) for word in text])

	# 把数组转为文本
	def arr_to_text(self,arr):
		return ''.join([self.vocab[i] if i < len(self.vocab) else '<unk>'for i in arr])

	# 把字典保存起来
	def save_to_file(self,filename):
		with open(filename,'wb') as file:
			pickle.dump(self.vocab,file)
	