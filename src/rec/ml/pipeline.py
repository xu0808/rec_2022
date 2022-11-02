# -*- coding:utf-8 -*-
# GBDT+LR
# https://www.cnblogs.com/siriJR/p/12180019.html
# 4.4 GBDT+LR模型训练及线上部署
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
# sklearn2pmml-0.86.0
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.ensemble import GBDTLRClassifier
from sklearn2pmml.pipeline import PMMLPipeline
import pandas
import os
import warnings
warnings.filterwarnings('ignore')


# 数据文件
data_dir = 'C:\\work\\data\\rec\\ml'
df = pandas.read_csv(os.path.join(data_dir, 'audit.csv'))
# cols = "Age,Employment,Education,Marital,Occupation,Income,Gender,Deductions,Hours,Adjusted"
print('cols:', list(df.columns))
# values = "38,Private,College,Unmarried,Service,81838.0,Female,False,72,0"
print('first line:', df.values[0, :])
# 数据列
cat_columns = ['Education', 'Employment', 'Marital', 'Occupation']
cont_columns = ['Age', 'Hours', 'Income']
label_column = 'Adjusted'

# 数据转换
mapper = DataFrameMapper(
    [([cat_column], [CategoricalDomain(), LabelBinarizer()]) for cat_column in cat_columns] +
    [(cont_columns, ContinuousDomain())]
)

# GBDT进行处理交叉
gbdt = GradientBoostingClassifier(n_estimators=499, max_depth=2)
lr = LogisticRegression()
classifier = GBDTLRClassifier(gbdt, lr)
pipeline = PMMLPipeline([
    ('mapper', mapper),
    ('classifier', classifier)
])

# 模型训练
pipeline.fit(df[cat_columns + cont_columns], df[label_column])
# 模型导出
sklearn2pmml(pipeline, os.path.join(data_dir, 'GBDT+LR.pmml'))
