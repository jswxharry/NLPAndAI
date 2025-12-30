# -*-coding: utf-8 -*-
# 对txt文件进行中文分词
import jieba
import os
from utils import files_processing

# 字词分割，对整个文件内容进行字词分割
def segment_lines(file_list,segment_out_dir,stopwords=[]):
    for i,file in enumerate(file_list):
        segment_out_name=os.path.join(segment_out_dir,'segment_{}.txt'.format(i))
        with open(file, 'rb') as f:
            document = f.read()
            document_cut = jieba.cut(document)
            sentence_segment=[]
            for word in document_cut:
                if word not in stopwords:
                    sentence_segment.append(word)
            result = ' '.join(sentence_segment)
            result = result.encode('utf-8')
            with open(segment_out_name, 'wb') as f2:
                f2.write(result)


# 三国源文件所在目录
source_folder = './Embeddings/word2vec/three_kingdoms/source'
segment_folder = './Embeddings/word2vec/three_kingdoms/segment'

file_list=files_processing.get_files_list(source_folder, postfix='*.txt')

# 打印看下文件是否被加载到了
print("待分词的文件列表：")
print(file_list)

#分词
segment_lines(file_list, segment_folder)
print("分词完成，分词文件保存在：{}".format(segment_folder))