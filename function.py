import numpy as np 

"""
说白了这个方法就是从前top_n个下标中带概率地随机选择一个下标，而每一个下标又对应一个字符
"""
def pick_top_n(preds, vocab_size, top_n = 5):
	# np.squeeze（）函数可以删除数组形状中的单维度条目，即把shape中为1的维度去掉
	p = np.squeeze(preds)
	# argsort函数返回的是数组值从小到大的索引值
	p[np.argsort(p)[:-top_n]] = 0
	# 归一化
	p = p / np.sum(p)
	# 随机选择一个字符
	# 从vocab_size中以概率p随机选择一个字符
	c = np.random.choice(vocab_size, 1, p = p)[0]
	return c