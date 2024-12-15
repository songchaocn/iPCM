import pandas as pd
import numpy as np

df = pd.read_csv('data/NYC/NYC_train.csv')
poi_df = pd.read_csv('data/NYC/poi_info.csv')
num_time_slices = 72
size_time_slices = 24 * 60 / num_time_slices
num_times = 20

cat_ids = list(set(poi_df['poi_catid'].tolist()))
cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))
cat_time = np.zeros((len(cat_ids), num_time_slices))
for i, row in df.iterrows():
    if(row['POI_catid'] in cat_id2idx_dict.keys()):
        c = cat_id2idx_dict[row['POI_catid']]
        t = int(int(row['UTC_time'][11:13]) * (60 / size_time_slices)) + int(int(row['UTC_time'][14:16]) / size_time_slices)
        cat_time[c][t] += 1

cat_timesum = np.sum(cat_time, axis=1)
cat_timesum = cat_timesum.argsort()[-50:][::-1]
time_cat = cat_time.T
time_cat_sum = np.sum(time_cat, axis=1)
for i in range(num_time_slices):
    time_cat[i] = time_cat[i] / time_cat_sum[i]

def get_class_ave(samples, start, end):
    class_ave = [0.0 for _ in range(len(samples[0]))]
    class_ave = np.array(class_ave)
    for i in range(start, end):
        class_ave += samples[i]
    class_ave = class_ave / (end - start)
    return class_ave

def get_class_diameter(samples, start, end):
    class_diameter = 0.0
    class_ave = get_class_ave(samples, start, end)
    for i in range(start, end):
        tem = samples[i] - class_ave
        for each in tem:
            class_diameter += each * each
    return class_diameter
    
def get_split_loss(samples, sample_num, split_class_num):
    split_loss_result = np.zeros((sample_num + 1, split_class_num + 1))
    split_loss_result1 = np.zeros((sample_num + 1, split_class_num + 1),dtype=int)

    for n in range(1, sample_num + 1):
        split_loss_result[n, 1] = get_class_diameter(samples, 0, n)

    for k in range(2, split_class_num + 1):
        for n in range(k, sample_num + 1):
            loss = []
            mi = 10000
            flag = k - 1
            for j in range(k - 1, n):
                if(split_loss_result[j, k - 1] + get_class_diameter(samples, j, n) < mi):
                    flag = j
                    mi = split_loss_result[j, k - 1] + get_class_diameter(samples, j, n)
            split_loss_result[n, k] = mi
            split_loss_result1[n, k] = flag

    return split_loss_result, split_loss_result1
split_loss_result, s1 = get_split_loss(time_cat, len(time_cat), len(time_cat))

num = num_times
last = num_time_slices
roots = []
while(num > 1):
    roots.append(s1[last][num])
    last = s1[last][num]
    num -= 1
roots = roots[::-1]
roots = [0] + roots + [num_time_slices]
time2idx_dic = {}
for i in range(len(roots)-1):
    left = roots[i]
    right = roots[i+1]
    for j in range(left, right):
        time2idx_dic[j] = (roots[i] + roots[i+1]) / (2.0 * num_time_slices)

df = pd.read_csv('data/NYC/NYC_train.csv')
for i in range(df.shape[0]):
    df.loc[i, 'time_period'] = time2idx_dic[int(df.loc[i, 'UTC_time'][11:13]) * (60 / size_time_slices) + int(int(df.loc[i, 'UTC_time'][14:16]) / size_time_slices)]
df.to_csv('data/NYC/NYC_train.csv', index=False)

df = pd.read_csv('data/NYC/NYC_val.csv')
for i in range(df.shape[0]):
    df.loc[i, 'time_period'] = time2idx_dic[int(df.loc[i, 'UTC_time'][11:13]) * (60 / size_time_slices) + int(int(df.loc[i, 'UTC_time'][14:16]) / size_time_slices)]
df.to_csv('data/NYC/NYC_val.csv', index=False)

df = pd.read_csv('data/NYC/NYC_test.csv')
for i in range(df.shape[0]):
    df.loc[i, 'time_period'] = time2idx_dic[int(df.loc[i, 'UTC_time'][11:13]) * (60 / size_time_slices) + int(int(df.loc[i, 'UTC_time'][14:16]) / size_time_slices)]
df.to_csv('data/NYC/NYC_test.csv', index=False)
