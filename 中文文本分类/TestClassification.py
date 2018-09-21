# _*_ coding:utf-8 _*_

import sys
import os
import jieba


# 中文语料分词

# 配置输出环境utf-8
reload(sys)
sys.setdefaultencoding('utf-8')

def savefile(savepath,content):  # 保存至文件
    fp = open(savepath,"wb")
    fp.write(content)
    fp.close()

def readfile(path):
    fp = open(path,"rb")
    content = fp.read()
    fp.close()
    return content


corpus_path = "E:\\Machine learning\\TextClassification\\train_corpus_small\\"   # 未分词分类语料库路径
seg_path = "E:\\Machine learning\\TextClassification\\train_corpus_seg\\" # 分词后分类语料库路径

catelist = os.listdir(corpus_path) # 获取corpus_path 下的所有子目录

# 获取每个目录下的所有文件
for mydir in catelist:
    class_path = corpus_path+mydir+"/"   # 拼出分类子目录的路径
    seg_dir = seg_path+mydir+"/"   # 拼出分词后的语料分类目录
    if not os.path.exists(seg_dir):   # 是否存在目录，若 不存在则创建
        os.makedirs(seg_dir)
    file_list = os.listdir(class_path)   # 获取类别目录下的所有文件
    for file_path in file_list:          # 遍历类别目录下的文件
        fullname = class_path+file_path  # 拼出文件名全路径
        content = readfile(fullname).strip() # 读取文件内容
        content = content.replace("\r\n","").strip()  # 删除换行和多余空格
        content_seg = jieba.cut(content)
        # 将处理好的文件保存到分词后语料目录
        savefile(seg_dir+file_path," ".join(content_seg))

print "中文语料分词结束"


# 文本向量信息对象化

from sklearn.datasets.base import Bunch # 导入Bunch类
import pickle

# Bunch类提供 一种key,value的对象 形式
# target_name:所有分类集名称列表
# lable:每个文件的分类标签列表
# filenames:文件路径
# contents:分词后文件词向量形式
bunch = Bunch(target_name=[],label=[],filename=[],contents=[])

# 分词语料Bunch对象持久化文件路径
wordbag_path = "E:\\Machine learning\\TextClassification\\train_word_bag\\train_set.dat"
# 分词后分类语料库路径
seg_path = "E:\\Machine learning\\TextClassification\\train_corpus_seg\\"

catelist = os.listdir(seg_path)
bunch.target_name.extend(catelist)  # 将类别信息保存到Bunch对象中
for mydir in catelist:
    class_path = seg_path+mydir+"/"
    file_list = os.listdir(class_path)   # 获取类别中文件
    for file_path in file_list:   # 遍历文件
        fullname = class_path + file_path   # 拼出文件全路径
        bunch.label.append(mydir)   # 保存当前文件的分类标签
        bunch.filename.append(fullname)  # 保存当前文件的文件路径
        bunch.contents.append(readfile(fullname).strip())  # 保存文件词向量

# Bunch 对象持久化
file_obj = open(wordbag_path,"wb")
pickle.dump(bunch,file_obj)
file_obj.close()

print "构件文本对象结束!!!"



# tf  idf

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer  # TF-IDF向量转换类
from sklearn.feature_extraction.text import TfidfVectorizer   # TF-IDF 向量生成类

# 读取Bunch对象
def readbunchobj(path):
    file_obj = open(path,"rb")
    bunch = pickle.load(file_obj)
    file_obj.close()
    return bunch

# 写入Bunch对象
def writebunchobj(path,bunchobj):
    file_obj = open(path,"wb")
    pickle.dump(bunchobj,file_obj)
    file_obj.close()

# 2. 导入分词后的词向量Bunch对象
path = "E:\\Machine learning\\TextClassification\\train_word_bag\\train_set.dat"  # 训练集词向量空间保存路径
bunch = readbunchobj(path)

# 3. 构建TF-IDF词向量空间对象
tfidfspace = Bunch(target_name=bunch.target_name,label=bunch.label,filename=bunch.filename,tdm=[],vocabulary={})

# 读取停用词表
stpwrdlst = readfile("E:\Machine learning\TextClassification\stop_words.txt").splitlines()


# 4. 使用TfidfVectorizer初始化向量空间模型
vectorizer = TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5)
transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
# 文本转为词频矩阵，单独保存字典文件
# 将词向量转换为词频矩阵-》Tf-idf加权文档项矩阵。
tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)   # 	学习词汇和idf，返回术语 - 文档矩阵。
tfidfspace.vocabulary = vectorizer.vocabulary_  #

# 5. 创建词袋的持久化
space_path = "E:\\Machine learning\\TextClassification\\train_word_bag\\tfidfspace.dat"  # 词向量词袋保存路径
writebunchobj(space_path,tfidfspace)

print "训练集TF-IDF计算完成！"



# 构建测试集向量模型

corpus_path = "E:\\Machine learning\\TextClassification\\test_corpus\\"   # 测试集未分词分类语料库路径
seg_path = "E:\\Machine learning\\TextClassification\\test_corpus_seg\\" # 测试集分词后分类语料库路径

catelist = os.listdir(corpus_path) # 获取corpus_path 下的所有子目录

# 获取每个目录下的所有文件
for mydir in catelist:
    class_path = corpus_path+mydir+"/"   # 拼出分类子目录的路径
    seg_dir = seg_path+mydir+"/"   # 拼出分词后的语料分类目录
    if not os.path.exists(seg_dir):   # 是否存在目录，若 不存在则创建
        os.makedirs(seg_dir)
    file_list = os.listdir(class_path)   # 获取类别目录下的所有文件
    for file_path in file_list:          # 遍历类别目录下的文件
        fullname = class_path+file_path  # 拼出文件名全路径
        content = readfile(fullname).strip() # 读取文件内容
        content = content.replace("\r\n","").strip()  # 删除换行和多余空格
        content_seg = jieba.cut(content)
        # 将处理好的文件保存到分词后语料目录
        savefile(seg_dir+file_path," ".join(content_seg))

print "测试集中文语料分词结束"


# 文本向量信息对象化

from sklearn.datasets.base import Bunch # 导入Bunch类
import pickle

# Bunch类提供 一种key,value的对象 形式
# target_name:所有分类集名称列表
# lable:每个文件的分类标签列表
# filenames:文件路径
# contents:分词后文件词向量形式
bunch = Bunch(target_name=[],label=[],filename=[],contents=[])

# 分词语料Bunch对象持久化文件路径
wordbag_path = "E:\\Machine learning\\TextClassification\\test_word_bag\\test_set.dat"
# 分词后分类语料库路径
seg_path = "E:\\Machine learning\\TextClassification\\test_corpus_seg\\"

catelist = os.listdir(seg_path)
bunch.target_name.extend(catelist)  # 将类别信息保存到Bunch对象中
for mydir in catelist:
    class_path = seg_path+mydir+"/"
    file_list = os.listdir(class_path)   # 获取类别中文件
    for file_path in file_list:   # 遍历文件
        fullname = class_path + file_path   # 拼出文件全路径
        bunch.label.append(mydir)   # 保存当前文件的分类标签
        bunch.filename.append(fullname)  # 保存当前文件的文件路径
        bunch.contents.append(readfile(fullname).strip())  # 保存文件词向量

# Bunch 对象持久化
file_obj = open(wordbag_path,"wb")
pickle.dump(bunch,file_obj)
file_obj.close()

print "构件测试集文本对象结束!!!"



# 导入分词后的词向量Bunch对象
path = "E:\\Machine learning\\TextClassification\\test_word_bag\\test_set.dat"  # 训练集词向量空间保存路径
bunch = readbunchobj(path)

# 3. 构建TF-IDF词向量空间对象
testspace = Bunch(target_name=bunch.target_name,label=bunch.label,filename=bunch.filename,tdm=[],vocabulary={})

# 导入训练集的词袋
trainbunch = readbunchobj("E:\\Machine learning\\TextClassification\\train_word_bag\\tfidfspace.dat")


# 4. 使用TfidfVectorizer初始化向量空间模型
vectorizer = TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5,vocabulary=trainbunch.vocabulary)  # 使用训练集词袋向量

transformer = TfidfTransformer()  # 该类会统计每个词语的TF-IDF权值
# 文本转为词频矩阵，单独保存字典文件
# 将词向量转换为词频矩阵-》Tf-idf加权文档项矩阵。
testspace.tdm = vectorizer.fit_transform(bunch.contents)   # 	学习词汇和idf，返回术语 - 文档矩阵。
testspace.vocabulary = vectorizer.vocabulary_  #

# 5. 创建词袋的持久化
space_path = "E:\\Machine learning\\TextClassification\\test_word_bag\\testspace.dat"  # 词向量词袋保存路径
writebunchobj(space_path,testspace)

print "测试集TF-IDF计算完成！"


from sklearn.naive_bayes import MultinomialNB # 导入多项式贝叶斯算法包

# 导入训练集向量空间
trainpath = "E:\\Machine learning\\TextClassification\\train_word_bag\\tfidfspace.dat"
train_set = readbunchobj(trainpath)

# 导入测试集向量空间
testpath = "E:\\Machine learning\\TextClassification\\test_word_bag\\testspace.dat"
test_set = readbunchobj(testpath)

# 应用朴素贝叶斯算法
# alpha:0.001 alpha越小，迭代次数越多，精度越高
clf = MultinomialNB(alpha=0.001).fit(train_set.tdm,train_set.label)

# 预测分类结果
predicted = clf.predict(test_set.tdm)
total = len(predicted);rate=0
for flabel,file_name,expct_cate in zip(test_set.label,test_set.filename,predicted):
    if flabel != expct_cate:
        rate += 1
        print file_name,":实际类别：",flabel," -->预测类别：",expct_cate

# 精度
print "error rate:",float(rate)*100/float(total),"%"
