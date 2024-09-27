import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os
from sklearn.neural_network import MLPClassifier

# 加载数据集
X = []
y = []
for expression in os.listdir('dataset'):
    for file in os.listdir(f'dataset/{expression}'):
        landmarks = np.load(f'dataset/{expression}/{file}')
        X.append(landmarks.flatten())
        y.append(expression)

# 打印分类的数量和标签
labels, counts = np.unique(y, return_counts=True)
print(f'Number of classes: {len(labels)}')
print(f'Labels: {labels}')
print(f'Counts: {counts}')

# 预处理数据
X = np.array(X)
y = np.array(y)
scaler = StandardScaler()
#X = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型
#clf = svm.SVC()
clf= MLPClassifier(hidden_layer_sizes=(100,), max_iter=300,activation='relu',solver='adam', random_state=21)

# 训练模型
clf.fit(X_train, y_train)

# 评估模型
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# 保存模型
from joblib import dump
dump(clf, 'bp_model.joblib') 
#dump(scaler, 'scaler.joblib') 