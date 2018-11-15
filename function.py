import numpy as np 
import copy

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


"""
批生成器 这里arr是整个训练集和字符串转为array的结果
n_seqs是指定的训练序列的数量
n_steps就是max_time
"""
def batch_generator(arr, n_seqs, n_steps):
    import copy
    arr = copy.copy(arr)
    batch_size = n_seqs * n_steps # batch_size为指定的序列数量 * 序列长度
    n_batches = int(len(arr)/batch_size) # 批的数量
    arr = arr[:batch_size * n_batches] # 这里是为了取整，如果有多余的就舍去了
    arr = arr.reshape(n_seqs,-1) #把真个训练数据变成 高为 n_seqs, 宽为 n_steps * n_batches 的矩阵
    while True:
        np.random.shuffle(arr) #把arr 按行打乱
        # 把这里想象成 切带鱼 因为宽为 n_steps * n_batches 所以一共切 n_batches次
        for n in range(0,arr.shape[1], n_steps):
            x = arr[:, n:n+ n_steps]
            y = np.zeros_like(x)
            #这行代码的意思是，假设一个字x，它后面的那个字就是y。就是说y是x的目标.
            y[:,:-1],y[:,-1] = x[:, 1:],x[:,0] # y 获取 x 的值，但是 x的开头需要放到最后再赋值给y
            yield x, y