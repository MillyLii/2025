# 第一个单元格 - 导入必要的库
import torch
import time
import os
import pickle
from CLEAN.dataloader import *
from CLEAN.model import *
from CLEAN.utils import *
import torch.nn as nn
from CLEAN.distance_map import get_dist_map
import torch.nn.functional as F
import random
from CLEAN.infer import *
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, cohen_kappa_score, matthews_corrcoef

def format_esm2(a):
    if type(a) == dict:
        # 检查可用的层
        if 33 in a['mean_representations']:
            return a['mean_representations'][33]
        # 如果33层不存在，使用最后一层
        last_layer = max(a['mean_representations'].keys())
        return a['mean_representations'][last_layer]
    return a

# 读取酶数据
enz_path = '/nfs/hb236/dhy/app/data/esm_data_apm'
enz_files = [os.path.join(enz_path, f) for f in os.listdir(enz_path) if f.endswith('.pt')]
enz_test2 = []
valid_files = []  # 存储成功加载的文件路径

print(f"正在读取amp数据，共有{len(enz_files)}个文件...")
for file in enz_files:
    try:
        data = torch.load(file)
        # 使用format_esm2函数处理数据
        vector = format_esm2(data)
        enz_test2.append(vector)
        valid_files.append(file)  # 只添加成功加载的文件
    except Exception as e:
        print(f"读取文件 {file} 时出错: {e}")

enz_test2 = torch.stack(enz_test2) if enz_test2 else torch.tensor([])
# enz_labels = torch.ones(enz_test2.shape[0])

# 从本地加载XGBoost模型
import xgboost as xgb
import os
loaded_model = xgb.XGBClassifier()

save_dir = './'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
model_path = os.path.join(save_dir, 'xgboost_model.json')
loaded_model.load_model(model_path)
print(f"模型已从 {model_path} 加载")

# 预测结果和概率
Enzo_results = loaded_model.predict(enz_test2)
Enzo_proba = loaded_model.predict_proba(enz_test2)[:, 1]

# 输出预测结果
for i, (result, prob, file_path) in enumerate(zip(Enzo_results, Enzo_proba, valid_files)):
    # 从文件路径中提取文件名作为序列ID
    file_name = os.path.basename(file_path).replace('.pt', '')
    if result == 0:
        print(f"序列 {file_name}: 预测为非酶，概率为 {(1-prob):.4f}")
    else:
        print(f"序列 {file_name}: 预测为酶，概率为 {prob:.4f}")

