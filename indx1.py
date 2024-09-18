# 导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud

# 读取数据集
dataset = pd.read_csv("mbti_1.csv")

# 创建二进制特征
dataset["I"] = dataset["type"].apply(lambda x: 1 if "I" in x else 0)
dataset["N"] = dataset["type"].apply(lambda x: 1 if "N" in x else 0)
dataset["T"] = dataset["type"].apply(lambda x: 1 if "T" in x else 0)
dataset["P"] = dataset["type"].apply(lambda x: 1 if "P" in x else 0)

# 设置主题
sns.set_theme()

# 去除URL的函数
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# 应用去除URL的函数
dataset["modified_post"] = dataset["posts"].apply(remove_urls)

# 停用词处理
all_stopwords = stopwords.words("english")
all_stopwords.remove("not")

# 文本预处理函数
def preprocess_text(phrase):
    phrase = re.sub('[^a-zA-Z]', ' ', phrase)
    phrase = phrase.lower().split()
    phrase = [word for word in phrase if word not in all_stopwords]
    return " ".join(phrase)

# 随机抽样数据集
mini_dataset = dataset.sample(n=2000)
mini_dataset.reset_index(inplace=True)

# 预处理文本
corpus = mini_dataset["modified_post"].apply(preprocess_text)

# 提取特征
ie = mini_dataset["I"]
ns = mini_dataset["N"]
tf = mini_dataset["T"]
pj = mini_dataset["P"]

# 特征向量化
cv = CountVectorizer(max_features=2500)
corpus = cv.fit_transform(corpus).toarray()

# 预测个性特征的函数
def predict_personality(y_train):
    sm = SMOTE(random_state=42)  # 处理不平衡数据
    X_train, X_test, y_train, y_test = train_test_split(corpus, y_train, test_size=0.2, random_state=42)

    # 输出训练集和测试集的标签分布
    print("Before OverSampling,... counts of label '1': {}".format(sum(y_train == 1)))
    print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

    # 进行过采样
    X_train, y_train = sm.fit_resample(X_train, y_train)

    print("After OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
    print("After OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

    # 使用随机森林分类器
    rf = RandomForestClassifier(n_estimators=100, random_state=42)  # 100棵树
    rf.fit(X_train, y_train)  # 训练模型

    # 预测训练集和测试集
    y_pred_train = rf.predict(X_train)
    print("Training Set accuracy: ", accuracy_score(y_pred_train, y_train))
    
    y_pred_test = rf.predict(X_test)
    print("Test set accuracy: ", accuracy_score(y_pred_test, y_test))

    # 计算精确率、召回率和F1分数
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    return confusion_matrix(y_pred_test, y_test) / len(y_pred_test), rf.feature_importances_

# 绘制混淆矩阵
f, axs = plt.subplots(4, 1, figsize=(20, 20))
sets = {"I/E": ie, "N/S": ns, "T/F": tf, "P/J": pj}
i = 0
for name, set in sets.items():
    print("Trait {}\n".format(name))
    cm, feature_importances = predict_personality(set)
    _ = sns.heatmap(cm, ax=axs[i], annot=True, cmap="crest", fmt='.2%')
    _.set_xlabel(name)
    print("\n")
    i += 1

# 特征重要性可视化
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(feature_importances)), feature_importances, align='center')
plt.yticks(range(len(feature_importances)), cv.get_feature_names_out())
plt.xlabel("Importance")
plt.show()

# 生成词云的函数
def word_cloud(mbti):
    mbti_texts = ""
    mbti_list = mini_dataset[mini_dataset["type"] == mbti]
    for text in mbti_list["modified_post"]:
        mbti_texts += text + " "
    wordcloud = WordCloud(max_font_size=50, max_words=150, background_color="white").generate(mbti_texts)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(mbti)
    plt.show()

# 生成特定MBTI类型的词云
mbtis = ["INTJ", "INFJ", "ENTP", "ENFP", "ISTJ", "ISFJ", "ESTP", "ESFP", "INTP", "ISTP", "ENTJ", "ESTJ", "INFP", "ISFP", "ESFJ", "ENFJ"]
for mbti in mbtis[:3]:  # 生成前3个MBTI类型的词云
    word_cloud(mbti)