import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False
data = pd.read_csv('D:/数据分析项目/金融风控之贷款违约/原始资料/train.csv')


# 1.查看数据维度、类型、统计量,了解特征含义
print(data.info())
print(data.describe())
"""
Field	            Description	          数据类型
id	                贷款清单的唯一标识	      无实际含义，考虑删除
loanAmnt	        贷款金额	              数值型
term	            贷款期限（year）	      数值型
interestRate	    贷款利率	              数值型
installment	        分期付款金额	          数值型
grade	            贷款等级	              有序类别型
subGrade	        贷款等级之子级	          有序类别型
employmentTitle	    职业名称	              无序类别型
employmentLength	就业年限（年）	          数值型
homeOwnership	    房屋所有权状况	          数值型
annualIncome	    年收入	              数值型
verificationStatus	验证状态	              无序类别型
issueDate	        贷款发放的时间	          时间特征
purpose	            贷款用途类别	          无序类别型
postCode	        邮政编码的前3位数字	      无序类别型
regionCode	        地区编码	              无序类别型
dti	                债务收入比	          数值型
delinquency_2years	借款人过去2年信用档案中逾期30天以上的违约事件数	        数值型
ficoRangeLow	    借款人在贷款发放时的fico信用分所属的下限范围	        数值型
ficoRangeHigh	    借款人在贷款发放时的fico信用分所属的上限范围            数值型
openAcc	            借款人信用档案中未结信用案件的数量	            数值型
pubRec	            贬损公共记录的数量	                        数值型
pubRecBankruptcies	公开记录清除的数量	                        数值型
revolBal	        信贷周转余额合计	                        数值型
revolUtil	        循环额度利用率，借款人使用的相对于所有可用循环信贷的信贷金额    数值型
totalAcc	        信用案件总数	                                        数值型
initialListStatus	贷款的初始列表状态	                                    无序类别型
applicationType	    贷款申请类型	                                        无序类别型
earliesCreditLine	借款人最早报告的信用额度开立的月份                        	时间型
title	            贷款名称	                                            无序类别型
policyCode	        公开可用的策略_代码=1新产品不公开可用的策略_代码=2	        无序类别型
n0-n14，            匿名特征，贷款人行为计数特征	                            数值型
isDefault           是否违约，1表示是，0表示否                               标签
"""
for i in ['id', 'employmentTitle', 'verificationStatus', 'purpose',
          'postCode', 'regionCode', 'initialListStatus', 'applicationType', 'title', 'policyCode']:
    data[i] = data[i].astype(str)

# 划分数据类型
# 数值型和类别型数据划分
numerical_feature = list(data.select_dtypes(exclude='object').columns)
category_feature = list(data.select_dtypes(include='object').columns)
label = 'isDefault'
numerical_feature.remove(label)


# 数值型划分连续变量和离散变量
def get_numerical_serial_features(df, feas):
    numerical_serial_feature=[]
    numerical_noserial_feature=[]
    for fea in feas:
        temp = df[fea].nunique()
        if temp <= 10:
            numerical_noserial_feature.append(fea)
        else:
            numerical_serial_feature.append(fea)
    return numerical_serial_feature, numerical_noserial_feature


numerical_serial_feature, numerical_noserial_feature = get_numerical_serial_features(data, numerical_feature)
print('查看连续变量', numerical_serial_feature)
print('查看离散变量', numerical_noserial_feature)


# 2.查看缺失值异常值
# 2.1查看缺失值
missing = data.isnull().sum()/len(data)  # 缺失率
missing = missing[missing > 0]  # 选择缺失率大于0
missing.sort_values(inplace=True)  # 排序
label = missing.index
plt.xticks(range(len(missing)), label)  # 设置刻度和标签
plt.bar(range(len(missing)), missing)  # 缺失率柱状图可视化
plt.show()

"""
缺失值比例最大不超过5%，先考虑均值填充，效果不好再考虑其它
"""

# 2.2箱型图查看异常值
f1 = pd.melt(data, value_vars=numerical_feature)  # 行列转换
g1 = sns.FacetGrid(f1, col="variable", col_wrap=4, sharex=False, sharey=False)  # 生成多个图
g1 = g1.map(sns.boxplot, 'value')
plt.show()
"""
异常值会影响模型泛化性能，确定错误的异常值删除，模型效果不好再考虑删除，替换
"""


# 3.查看特征分布
# 3.1连续特征分布可视化
f2 = pd.melt(data, value_vars=numerical_serial_feature)
g2 = sns.FacetGrid(f2, col="variable",  col_wrap=4, sharex=False, sharey=False)
g2 = g2.map(sns.distplot, "value", kde_kws={'bw': 1})
plt.xticks(rotation=90)
plt.show()

# 3.2离散特征分布可视化
f3 = pd.melt(data.loc[:, numerical_noserial_feature], value_vars=numerical_noserial_feature)
g3 = sns.FacetGrid(f3, col='variable', hue="variable", col_wrap=4, sharex=False, sharey=False)
g3 = g3.map(sns.countplot, 'value')
plt.show()
"""
结论：
n11,n12,policycode,applicationtype某一值占比超过90%，分布异常不平衡，可以考虑删除
"""

# 3.4类别特征分布
for i in category_feature:
    print(data[i].value_counts())
"""
结论：
低基数类别特征考虑独热编码，高基数无序类别型特征考虑均值编码，有序类别特征考虑标签编码或有序字典映射
时间特征考虑计算时间段，可能会有好的效果
"""


# 4.共线性分析
plt.figure(figsize=(16, 16), dpi=1500)
data_corr = data[numerical_feature].corr()
data_corr[data_corr <= 0.8] = 0.01
sns.heatmap(data_corr)  # 热力图
plt.show()
"""
结论：
存在相关性大于0.8的特征，线性模型需要考虑去除共线性
"""