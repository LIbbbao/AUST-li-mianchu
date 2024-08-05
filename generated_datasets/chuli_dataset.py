import pandas as pd 
import numpy as np  
from sklearn.preprocessing import StandardScaler  
import random  

# 读取CSV文件  
file_path = "/home/lijiabao/SAITS/generated_datasets/AUST_gait_prediction_dataset_train_data.csv"  
data = pd.read_csv(file_path)  

# 将字符串格式的特征列转换为数值数据  
def parse_features(data):  
    parsed_data = []  
    for index, row in data.iterrows():  
        parsed_row = [list(map(float, cell.split())) for cell in row[:-1]]  
        parsed_data.append(parsed_row)  
    return np.array(parsed_data)  

parsed_data = parse_features(data)  

# 提取状态标签  
state_labels = data['Gait category'].values  

# 确认数据尺寸  
num_samples, num_timesteps, num_features = parsed_data.shape  

# 数据标准化  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(parsed_data.reshape(num_samples, num_timesteps * num_features))  
X_scaled = X_scaled.reshape(num_samples, num_timesteps, num_features)  

# 创建mask并进行随机掩盖  
mask = np.ones_like(X_scaled, dtype=bool)  

# 随机掩盖10%的数据  
random.seed(0)  # 固定随机种子确保结果可重复  
for i in range(num_samples):  
    for j in range(num_timesteps):  
        for k in range(num_features):  
            if random.random() < 0.1:  # 10% 概率掩盖  
                mask[i, j, k] = False  
                X_scaled[i, j, k] = np.nan  

# 生成符合SAITS模型输入要求的数据  
saits_input = {  
    'X': X_scaled,  
    'missing_mask': mask,  
    'X_holdout': X_scaled.copy(),  # 这里假设X_holdout与X相同，实际情况可能不同
    'indicating_mask': ~mask,  
    'labels': state_labels  
}  

# 将预处理后的数据保存到本地文件  
np.savez('saits_input_data.npz', **saits_input)  

print("数据预处理完成并保存到 saits_input_data.npz")  

