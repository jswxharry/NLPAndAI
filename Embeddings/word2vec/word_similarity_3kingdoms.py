# -*-coding: utf-8 -*-
# 先运行 word_seg进行中文分词，然后再进行word_similarity计算
# 将Word转换成Vec，然后计算相似度 
from gensim.models import word2vec
import multiprocessing

# 如果报文件读取错误，跑以下命令检查你的当前目录，调整文件地址
import os
print("当前工作目录：")
print(os.getcwd())

# 如果目录中有多个文件，可以使用PathLineSentences
segment_folder = './Embeddings/word2vec/three_kingdoms/segment'
# 切分之后的句子合集
sentences = word2vec.PathLineSentences(segment_folder)

# 设置模型参数，进行训练
model = word2vec.Word2Vec(sentences, vector_size=100, window=3, min_count=1)

print("三国人物相似度测试：")
print("曹操 和 刘备 相似度：" + str(model.wv.similarity('曹操', '刘备')))
print("曹操 和 张飞 相似度：" + str(model.wv.similarity('曹操', '张飞')))
print("刘备 和 张飞 相似度：" + str(model.wv.similarity('刘备', '张飞')))
print("曹操 最相似的词：" + str(model.wv.most_similar(positive=['曹操'])))
print("刘备 最相似的词：" + str(model.wv.most_similar(positive=['刘备'])))
print("张飞 最相似的词：" + str(model.wv.most_similar(positive=['张飞'])))
print("曹操 + 刘备 - 张飞 最相似的词：" + str(model.wv.most_similar(positive=['曹操', '刘备'], negative=['张飞'])))

# 设置模型参数，进行训练
model2 = word2vec.Word2Vec(sentences, vector_size=160, window=10, min_count=8, workers=multiprocessing.cpu_count())
# 保存模型 可以先不保存
model2.save('./Embeddings/word2vec/models/word2Vec_3kindoms.model')
print("三国人物相似度测试：")
print("曹操 和 刘备 相似度：" + str(model2.wv.similarity('曹操', '刘备')))
print("曹操 和 张飞 相似度：" + str(model2.wv.similarity('曹操', '张飞')))
print("刘备 和 张飞 相似度：" + str(model2.wv.similarity('刘备', '张飞')))
print("曹操 最相似的词：" + str(model2.wv.most_similar(positive=['曹操'])))
print("刘备 最相似的词：" + str(model2.wv.most_similar(positive=['刘备'])))
print("张飞 最相似的词：" + str(model2.wv.most_similar(positive=['张飞'])))
print("曹操 + 刘备 - 张飞 最相似的词：" + str(model2.wv.most_similar(positive=['曹操', '刘备'], negative=['张飞'])))