import pandas as pd
import numpy as np
import time
import datetime
from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve  # 查看是否过拟合
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 50)
data = pd.read_csv('D:/数据分析项目/金融风控之贷款违约/原始资料/train.csv')

# 1数据清洗与预处理
# 1.1单位和特殊字符清洗


def clean_currency(x):
    # 如果是字符串，就进行清洗，否则就返回
    if isinstance(x, str):
        return x.replace('years', '').replace('year', '').replace('10+', '10').replace('< 1', '1')
    return x


data['employmentLength'] = data['employmentLength'].apply(clean_currency).astype(float)  # 缺失值不能转化成整形


# 1.2时间格式转化
data['issueDate'] = pd.to_datetime(data['issueDate'], format='%Y-%m-%d')
data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda x: time.strptime(x, '%b-%Y'))  # 字符串转化为时间格式
data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda x: time.strftime('%Y-%m-%d', x))  # 时间格式转化
data['earliesCreditLine'] = pd.to_datetime(data['earliesCreditLine'], format='%Y-%m-%d')  # 转化为datetime格式
print(data.info())

# 1.3数据类型转化，将类别型变量转化为对象型数据格式
for i in ['id', 'employmentTitle', 'verificationStatus', 'purpose',
          'postCode', 'regionCode', 'initialListStatus', 'applicationType', 'title', 'policyCode']:
    data[i] = data[i].astype(str)

# 1.4 删除无实际含义、唯一值的特征
data.drop(['id', 'policyCode'], axis=1, inplace=True)

# 1.5缺失值处理
# 1.5.1缺失值比例最大不大超过%，考虑均值填充，模型效果太好，再考虑删除或算法填充
for fea in category_feature:
    data[fea] = data[fea].fillna(data[fea].mode()[0])
# 1.5.2连续数据中位数填充
for fea in numerical_feature:
    data[fea] = data[fea].fillna(data[fea].median())
print(data.info())

# 1.6异常值处理
# 划分数值型和类别型数据
numerical_feature = list(data.select_dtypes(exclude=['object', 'datetime64']).columns)
category_feature = list(data.select_dtypes(include='object').columns)
label = 'isDefault'
numerical_feature.remove(label)


# 初步划分连续特征和离散特征
def get_numercial_serial_features(df, feas):
    # 取值个数大于10个的划分为连续特征，否则划分为离散特征
    numerical_serial_feature = []  # 创建空列表
    numerical_noserial_feature = []
    for fea in feas:
        temp = df[fea].nunique()
        if temp <= 10:
            numerical_noserial_feature.append(fea)
        else:
            numerical_serial_feature.append(fea)
    return numerical_serial_feature, numerical_noserial_feature


numerical_serial_feature, numerical_noserial_feature = get_numercial_serial_features(data, numerical_feature)

# 划分数据集，测试集不处理异常值#，分层抽样，保证训练集测试集数据分布一致
train, test = train_test_split(data, test_size=0.2, stratify=data.loc[:, 'isDefault'], random_state=2022)
# 异常值删除,债务收入比不可能是一个负值，删除
train = train[train['dti'] >= 0]
train_processed = train
test_processed = test  # 测试集不处理异常值
print(train_processed.info(), test_processed.info())

# 2特征工程
# 2.1特征编码
# 2.1.1有序字典映射（标签编码）
for da in [train_processed, test_processed]:
    da['grade'] = da['grade'].map({'A': 0, 'B': 5, 'C': 10, 'D': 15, 'E': 20, 'F': 25, 'G': 30})
    da['subGrade'] = da['subGrade'].map({'A1': 0, 'B1': 5, 'C1': 10, 'D1': 15, 'E1': 20, 'F1': 25, 'G1': 30,
                                         'A2': 1, 'B2': 6, 'C2': 11, 'D2': 16, 'E2': 21, 'F2': 26, 'G2': 31,
                                         'A3': 2, 'B3': 7, 'C3': 12, 'D3': 17, 'E3': 22, 'F3': 27, 'G3': 32,
                                         'A4': 3, 'B4': 8, 'C4': 13, 'D4': 18, 'E4': 23, 'F4': 28, 'G4': 33,
                                         'A5': 4, 'B5': 9, 'C5': 14, 'D5': 19, 'E5': 24, 'F5': 29, 'G5': 34})
# 2.2.2高基数类特征均值编码
train_data = train_processed
train_y = train_processed['isDefault']
encoder = TargetEncoder(cols=['employmentTitle', 'purpose', 'postCode', 'regionCode'],
                        handle_unknown='value',
                        handle_missing='value').fit(train_data, train_y)  # 在训练集上训练
encoded_train = encoder.transform(train_data)  # 转换训练集
encoded_test = encoder.transform(test_processed)  # 转换测试集
print('完成特征编码,查看数据类型', encoded_train.info())

# 2.3特征构造（衍生）
# 2.3.1计算每个等级内其它连续特征的均值
for df in [encoded_train, encoded_test]:
    for item in numerical_serial_feature:
        df['grade_to_mean_' + item] = df.groupby(['grade'])[item].transform('mean')

# 2.3.2 部分连续特征的相除
for df in [encoded_train, encoded_test]:
    # 还款总金额与周转余额比=分期付款金额 * 12 * 贷款期限/信贷周转余额合计
    df['installment_term_revolBal'] = df['installment'] * 12 * df['term'] / (df['revolBal'] + 0.1)
    # 可用信贷额度与周转余额比=循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额/信贷周转余额合计
    df['revolUtil_revolBal'] = df['revolUtil'] / (df['revolBal'] + 0.1)
    # 信贷周转余额合计/贷款金额
    df['revolUtil_revolBal'] = df['revolBal'] / (df['loanAmnt'] + 0.1)
    # 未结信用数量与当前信用案件总数量=借款人信用档案中未结信用案件的数量/借款人信用档案中当前的信用案件总数
    df['openAcc_totalAcc'] = df['openAcc'] / df['totalAcc']
    # 贷款金额占个人全部债务金额比例=贷款金额/（债务收入比*年收入=总债务）
    df['loanAmnt_dti_annualIncome'] = df['loanAmnt'] / (np.abs(df['dti']) * df['annualIncome'] + 0.1)
    # 收入贷款金额比=年收入/贷款金额，值越大，可能还款压力越大，风险高
    df['annualIncome_loanAmnt'] = df['annualIncome'] / (df['loanAmnt'] + 0.1)
    # 信贷周转余额合计/分期付款金额
    df['revolBal_installment'] = df['revolBal'] / (df['installment'] + 0.1)
    # annualIncome_installment：年收入/分期付款金额
    df['annualIncome_installment'] = df['annualIncome'] / (df['installment'] + 0.1)


# 2.3.3时间特征衍生，计算时间段
for df in [encoded_train, encoded_test]:
    today = datetime.datetime.today()
    df['earliesCreditLine'] = pd.to_datetime(df['earliesCreditLine'], format='%Y-%m-%d')
    # 开户时间距离现在越久，可能说明信用越好
    df['today_earliesCreditLine'] = df['earliesCreditLine'].apply(lambda x: today-x).dt.days/30
    # 贷款发放时间距离现在越久，可能说明信还款风险越大
    df['issueDate'] = pd.to_datetime(df['issueDate'], format='%Y-%m-%d')
    df['today_issueDate'] = df['issueDate'].apply(lambda x: today-x).dt.days/30
    df.drop(['earliesCreditLine', 'issueDate'], axis=1, inplace=True)
print('完成特征构建')

# # 2.4特征筛选
# # 2.4.1决策树做基模型的筛选
x_cart = encoded_train.drop('isDefault', axis=1)
y_cart = encoded_train['isDefault']
# 基于树的训练
clf = DecisionTreeClassifier(criterion='gini')
clf = clf.fit(x_cart, y_cart)
# 特征重要性(数值越高特征越重要)
model_select = SelectFromModel(clf, threshold='mean', prefit=True)  # 选择特征重要性大于均值的特征
model_select.transform(x_cart)
model_select.get_support(True)  # 返回所选的特征索引
fea_list = []
for i in model_select.get_support(True):
    fea_list.append(x_cart.columns[i])
fea_list.append('isDefault')

train_selected = encoded_train[fea_list]  # # cart树筛选后的数据筛
test_selected = encoded_train[fea_list]  # 测试集与训练集特征保持一致
print('完成特征筛选', '查看数据')
print(train_selected.info(), test_selected.info())


# 3训练与调参
x_train = train_selected.drop('isDefault', axis=1)
y_train = train_selected['isDefault']
x_test = test_selected.drop('isDefault', axis=1)
y_test = test_selected['isDefault']
# 3.1算法选择
model = {}
model['rfc'] = RandomForestClassifier()
model['gdbt'] = GradientBoostingClassifier()
model['cart'] = DecisionTreeClassifier()
model['lr'] = LogisticRegression()
for i in model:
    model[i].fit(x_train, y_train)
    score = cross_val_score(model[i], x_train, y_train, cv=3, scoring='roc_auc')
    print('%s的auc为：%.3f' % (i, score.mean()))  # 保留三位小数

print('已完成算法选择')
# # 3.2调参
gbdt_adj = GradientBoostingClassifier(random_state=2022)
params = {'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
          'n_estimators': [25, 50, 100, 150, 200],
          'max_depth': [2, 3, 4, 5, 6],
          'min_samples_split': [50, 100, 200, 300, 400],  # 节点划分所需最小样本数
          'min_samples_leaf': [50, 100, 200, 300, 400]}  # 叶结点最小样本数
best = GridSearchCV(gbdt_adj, param_grid=params, refit=True, cv=3, scoring='roc_auc').fit(x_train, y_train)
print('best parameters:', best.best_params_)
print('已完成参数调优')

# 3.3模型训练
model_gbdt = GradientBoostingClassifier(learning_rate=0.15, max_depth=3, n_estimators=100, min_samples_split=100,
                                        min_samples_leaf=100, random_state=2022)
model_gbdt.fit(x_train, y_train)
print('完成模型训练')


# 4模型评估
# 4.1泛化能力，查看学习曲线
train_sizes, train_scores, test_scores = learning_curve(estimator=model_gbdt, X=x_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1, 10), cv=3,
                                                        scoring='roc_auc', n_jobs=-1)
# 统计结果
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
# 绘制效果
plt.plot(train_sizes, train_mean, color='red', marker='o', markersize=5, label='training auc')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test auc')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Auc')
plt.legend(loc='lower right')
plt.ylim([0.5, 1.0])  # 纵坐标起始值
plt.show()

# 4.2评估指标
y_pre = model_gbdt.predict(x_test)
# roc曲线
yscore = model_gbdt.predict_proba(x_test)
y_score = yscore[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)  # 假正率，真正率和阈值
# 绘制效果
plt.plot(fpr, tpr, color='brown', label="Test - AUC")  # 线图
plt.title("ROC curve")  # 标题
plt.xlabel("FPR")  # 横坐标标签
plt.ylabel("TPR")  # 纵坐标标签
plt.legend(labels=label, loc="lower right")  # 图例
plt.show()
# auc值
auc = roc_auc_score(y_test, y_score)
print('auc', auc)
# 4.3保存模型，使用sklearn中的模块joblib
joblib.dump(model_gbdt, 'D:/个人违约预测/model_gbdt.pkl')
print('已完成模型保存')
