import argparse
import datetime
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import KMeans
import random
from model import POIGraph, TransformerModel, UserEmbeddings, Time2Vec, FuseEmbeddings

device = torch.device('cpu')

parser = argparse.ArgumentParser()

parser.add_argument('--data_train', default='data/NYC/NYC_train.csv')
parser.add_argument('--data_val', default='data/NYC/NYC_val.csv')
parser.add_argument('--data_test', default='data/NYC/NYC_test.csv')
parser.add_argument('--data_poi_info', default='data/NYC/poi_info.csv')
parser.add_argument('--num_clusters', type=int, default=300, help='the number of clusters')
parser.add_argument('--time_feature', type=str, default='time_period')
parser.add_argument('--short_traj_thres',type=int, default=2, help='Remove over-short trajectory')
parser.add_argument('--batch', type=int, default=100, help='input batch size')
parser.add_argument('--poi_embed_dim', type=int, default=128, help='the dimension of POI embedding')
parser.add_argument('--user_embed_dim', type=int, default=128, help='the dimension of user embedding')
parser.add_argument('--region_embed_dim', type=int, default=64, help='the dimension of region embedding')
parser.add_argument('--time_embed_dim', type=int, default=32, help='the dimension of time embedding')
parser.add_argument('--epochs', type=int, default=100, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr-scheduler-factor', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
parser.add_argument('--device', type=str,default=device)
parser.add_argument('--transformer_nhead', type=int,default=2)
parser.add_argument('--transformer_nhid', type=int,default=1024)
parser.add_argument('--transformer_nlayers', type=int,default=2)
parser.add_argument('--transformer_dropout', type=float,default=0.3)
parser.add_argument('--patience', type=int, default=8, help='Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(args.seed)

train_df = pd.read_csv(args.data_train)
val_df = pd.read_csv(args.data_val)
test_df = pd.read_csv(args.data_test)
poi_df = pd.read_csv(args.data_poi_info)

# POI id to index
poi_ids = poi_df['poi_id'].tolist()
poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))
num_pois = len(poi_ids)

# cluster
poi2region_dic = {}
ldf = poi_df[['longitude', 'latitude']]
data = np.array(ldf)
labels = KMeans(n_clusters=args.num_clusters, max_iter=1000).fit_predict(data)
poi_df['label'] = labels
print("cluster completed")

# POI to region
for i, row in poi_df.iterrows():
    poi2region_dic[poi_id2idx_dict[row['poi_id']]] = int(row['label'])
num_regions = args.num_clusters

# User id to index
user_ids = [str(each) for each in list(set(train_df['user_id'].to_list()))]
user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))

# time to index
time_ids = [each for each in list(set(train_df['time_period'].to_list()))]
time_id2idx_dict = dict(zip(time_ids, range(len(time_ids))))
num_times = len(time_ids)

# User-POI count
num_users = len(user_ids)
user_poi = np.zeros((num_users, num_pois))
user_region_poi = np.zeros((num_users, num_regions, num_pois))
user_time_poi = np.zeros((num_users, num_times, num_pois))
users = list(set(train_df['user_id'].to_list()))
for user in users:
    user_df = train_df[train_df['user_id'] == user]
    user_df = user_df.reset_index(drop = True)
    user_idx = user_id2idx_dict[str(user)]
    for i, row in user_df.iterrows():
        poi_idx = poi_id2idx_dict[row['POI_id']]
        region_idx = poi2region_dic[poi_idx]
        user_poi[user_idx][poi_idx] += 1
        user_region_poi[user_idx][region_idx][poi_idx] = 1
        user_time_poi[user_idx][time_id2idx_dict[row['time_period']]][poi_idx] = 1
user_poi_sum = np.sum(user_poi, axis=1)
for i in range(user_poi.shape[0]):
    user_poi[i] = user_poi[i] / user_poi_sum[i]
user_poi = torch.from_numpy(user_poi)
user_poi = user_poi.to(device=args.device, dtype=torch.float)

edge_index = [[],[]]
edge_index1 = [[],[]]
for traj_id in set(train_df['trajectory_id'].tolist()):
    traj_df0 = train_df[train_df['trajectory_id'] == traj_id]
    traj_df0 = traj_df0.reset_index(drop=True)
    for i in range(traj_df0.shape[0] - 1):
        edge_index[0].append(poi_id2idx_dict[traj_df0.loc[i, 'POI_id']])
        edge_index[1].append(poi_id2idx_dict[traj_df0.loc[i + 1, 'POI_id']])
        edge_index1[0].append(poi2region_dic[poi_id2idx_dict[traj_df0.loc[i, 'POI_id']]])
        edge_index1[1].append(poi2region_dic[poi_id2idx_dict[traj_df0.loc[i + 1, 'POI_id']]])

class TrajectoryDatasetTrain(Dataset):
    def __init__(self, train_df):
        self.df = train_df
        self.traj_seqs = []
        self.input_seqs = []
        self.label_seqs = []

        for traj_id in tqdm(set(train_df['trajectory_id'].tolist())):
            traj_df0 = train_df[train_df['trajectory_id'] == traj_id]
            traj_df0 = traj_df0.reset_index(drop=True)
            if(traj_df0.shape[0] > 50):
                traj_df = traj_df0.iloc[-50:,:]
            else:
                traj_df = traj_df0
            poi_ids = traj_df['POI_id'].to_list()
            poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
            time_feature = traj_df[args.time_feature].to_list()
            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))
            if len(input_seq) < args.short_traj_thres:
                continue
            self.traj_seqs.append(traj_id)
            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])
    
class TrajectoryDatasetVal(Dataset):
    def __init__(self, df):
        self.df = df
        self.traj_seqs = []
        self.input_seqs = []
        self.label_seqs = []

        for traj_id in tqdm(set(df['trajectory_id'].tolist())):
            user_id = traj_id.split('_')[0]
            if user_id not in user_id2idx_dict.keys():
                continue
            traj_df0 = df[df['trajectory_id'] == traj_id]
            traj_df0 = traj_df0.reset_index(drop=True)
            if(traj_df0.shape[0] > 50):
                traj_df = traj_df0.iloc[-50:,:]
            else:
                traj_df = traj_df0
            poi_ids = traj_df['POI_id'].to_list()
            poi_idxs = []
            time_feature = traj_df[args.time_feature].to_list()
            time_idxs = []
            for each in range(len(poi_ids)):
                if poi_ids[each] in poi_id2idx_dict.keys():
                    poi_idxs.append(poi_id2idx_dict[poi_ids[each]])
                    time_idxs.append(time_feature[each])
                else:
                    continue
            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_idxs[i]))
                label_seq.append((poi_idxs[i + 1], time_idxs[i + 1]))
            if len(input_seq) < args.short_traj_thres:
                continue
            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)
            self.traj_seqs.append(traj_id)

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])
    
print('dataloader...')
train_dataset = TrajectoryDatasetTrain(train_df)
val_dataset = TrajectoryDatasetVal(val_df)
test_dataset = TrajectoryDatasetVal(test_df)
train_loader = DataLoader(train_dataset,
                            batch_size=args.batch,
                            shuffle=True,
                            pin_memory=True,
                            collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset,
                        batch_size=args.batch,
                        shuffle=False,
                        pin_memory=True,
                        collate_fn=lambda x: x)
test_loader = DataLoader(test_dataset,
                        batch_size=args.batch,
                        shuffle=False,
                        pin_memory=True,
                        collate_fn=lambda x: x)

poi_embed_model = POIGraph(num_pois, args.poi_embed_dim)
user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)
time_embed_model = Time2Vec(args.time_embed_dim)
region_embed_model = POIGraph(num_regions, args.region_embed_dim)
embed_fuse_model = FuseEmbeddings(args.poi_embed_dim + args.user_embed_dim, args.time_embed_dim + args.region_embed_dim)
dim = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.region_embed_dim
seq_model = TransformerModel(num_pois,
                             num_regions,
                             num_times,
                                dim,
                                args.transformer_nhead,
                                args.transformer_nhid,
                                args.transformer_nlayers,
                                dropout=args.transformer_dropout,
                            )

optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                list(user_embed_model.parameters()) +
                                list(time_embed_model.parameters()) +
                                list(embed_fuse_model.parameters()) +
                                list(seq_model.parameters()) +
                                list(region_embed_model.parameters()),
                        lr=args.lr,
                        weight_decay=args.weight_decay)

criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)
criterion_region = nn.CrossEntropyLoss(ignore_index=-1)
criterion_time = nn.CrossEntropyLoss(ignore_index=-1)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor, patience=args.patience)

def input_traj_to_embeddings(sample, poi_embeddings, user_embeddings, region_embeddings):
    traj_id = sample[0]
    input_seq = [each[0] for each in sample[1]]
    input_seq_time = [each[1] for each in sample[1]]

    user_id = traj_id.split('_')[0]
    user_idx = user_id2idx_dict[user_id]
    user_embedding = user_embeddings[user_idx]
    user_embedding = torch.squeeze(user_embedding).to(device=args.device)

    input_seq_embed = []
    for idx in range(len(input_seq)):
        poi_embedding = poi_embeddings[input_seq[idx]]
        poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)

        region_idx = poi2region_dic[input_seq[idx]]
        region_embedding = region_embeddings[region_idx]
        region_embedding = torch.squeeze(region_embedding)

        time_embedding = time_embed_model(torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
        time_embedding = torch.squeeze(time_embedding).to(device=args.device)

        c1 = torch.cat((user_embedding, poi_embedding), dim=-1)
        c2 = torch.cat((time_embedding, region_embedding), dim=-1)
        fused_embedding = embed_fuse_model(c1, c2)

        input_seq_embed.append(fused_embedding)
    return input_seq_embed

def top_k_acc(y_true_seq, y_pred_seq, k):
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    idx = np.where(top_k_rec == y_true)[0]
    if len(idx) != 0:
        return 1
    else:
        return 0
    
def MRR_metric(y_true_seq, y_pred_seq):
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)

def softmax(x):
    x -= np.max(x)
    x = np.exp(x) / np.sum(np.exp(x))
    return x

def adjust_pred_pro(y_pred_poi, y_pred_region, batch_seq_users, y_pred_time):
    y_pred_poi_adjusted = np.zeros_like(y_pred_poi)

    for i in range(len(batch_seq_lens)):
        useridx = batch_seq_users[i]
        traj_i_input = batch_seq_lens[i]
        j = traj_i_input - 1
        region_pro = y_pred_region[i][j]
        time_pro = y_pred_time[i][j]
        poi_pro = y_pred_poi[i, j, :]
        pro = np.zeros(num_pois)

        regions = region_pro.argsort()[-20:]
        times1 = time_pro.argsort()[-1]
        times2 = time_pro.argsort()[-2]
        times3 = time_pro.argsort()[-3]
        times4 = time_pro.argsort()[-4]
        times5 = time_pro.argsort()[-5]

        for k in range(num_pois):
            if((user_time_poi[useridx][times1][k] != 0 or user_time_poi[useridx][times2][k] != 0 or user_time_poi[useridx][times3][k] != 0
                    or user_time_poi[useridx][times4][k] != 0 or user_time_poi[useridx][times5][k] != 0) and poi2region_dic[k] in regions):
                pro[k] = 1.0
        poi_pro = softmax(poi_pro)
        y_pred_poi_adjusted[i, j, :] = pro + poi_pro

    return y_pred_poi_adjusted

poi_idxs = [i for i in range(num_pois)]
user_idxs = [i for i in range(num_users)]
region_idxs = [i for i in range(num_regions)]

poi_idxs = torch.Tensor(poi_idxs).long().to(device=args.device)
user_idxs = torch.Tensor(user_idxs).long().to(device=args.device)
region_idxs = torch.Tensor(region_idxs).long().to(device=args.device)

edge_index = torch.LongTensor(edge_index).to(device=args.device)
edge_index1 = torch.LongTensor(edge_index1).to(device=args.device)

poi_embed_model = poi_embed_model.to(device=args.device)
user_embed_model = user_embed_model.to(device=args.device)
time_embed_model = time_embed_model.to(device=args.device)
region_embed_model = region_embed_model.to(device=args.device)
embed_fuse_model = embed_fuse_model.to(device=args.device)
seq_model = seq_model.to(device=args.device)


best = 0
best_epoch = 0
best_epoch_test_loss = 0
best_epoch_test_top1_acc = 0
best_epoch_test_top5_acc = 0
best_epoch_test_top10_acc = 0
best_epoch_test_top20_acc = 0
best_epoch_test_mrr = 0

for epoch in range(args.epochs):
    print("**********epoch"+str(epoch)+"**********")
    print(datetime.datetime.now())
    poi_embed_model.train()
    user_embed_model.train()
    time_embed_model.train()
    embed_fuse_model.train()
    seq_model.train()
    region_embed_model.train()

    train_batches_top20_acc_list = []
    train_batches_mrr_list = []
    train_batches_loss_list = []

    for b_idx, batch in enumerate(train_loader):
        batch_pred_seqs = []
        batch_seq_lens = []
        batch_seq_embeds = []
        batch_seq_labels_poi = []
        batch_seq_labels_time = []
        batch_seq_labels_region = []
        batch_seq_users = []

        poi_embeddings = poi_embed_model(poi_idxs, edge_index)
        region_embeddings = region_embed_model(region_idxs, edge_index1)
        user_embeddings = user_embed_model(user_idxs, torch.mm(user_poi, poi_embeddings))

        for sample in batch:
            traj_id = sample[0]
            user_id = traj_id.split('_')[0]
            user_idx = user_id2idx_dict[user_id]
            input_seq_emb_list = []

            input_seq = [each[0] for each in sample[1]]
            label_seq = [each[0] for each in sample[2]]

            label_seq_time = [time_id2idx_dict[each[1]] for each in sample[2]]
            label_seq_region = [poi2region_dic[each] for each in label_seq]

            input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings, user_embeddings, region_embeddings))
            batch_seq_users.append(user_idx)
            batch_seq_embeds.append(input_seq_embed)
            batch_seq_lens.append(len(input_seq))
            batch_seq_labels_poi.append(torch.LongTensor(label_seq))
            batch_seq_labels_time.append(torch.LongTensor(label_seq_time))
            batch_seq_labels_region.append(torch.LongTensor(label_seq_region))

        batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
        label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
        label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
        label_padded_region = pad_sequence(batch_seq_labels_region, batch_first=True, padding_value=-1)

        mask = (torch.triu(torch.ones(batch_padded.size(1), batch_padded.size(1))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        src_mask = torch.zeros((batch_padded.size(0) * 2, batch_padded.size(1), batch_padded.size(1)))
        for i in range(src_mask.size(0)):
            src_mask[i] = mask
 
        src_key_padding_mask = torch.zeros((batch_padded.size(0), batch_padded.size(1)), dtype=torch.bool)
        for i in range(batch_padded.size(0)):
            for j in range(batch_seq_lens[i], batch_padded.size(1)):
                src_key_padding_mask[i][j] = True

        x = batch_padded.to(device=args.device, dtype=torch.float)
        y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
        y_time = label_padded_time.to(device=args.device, dtype=torch.long)
        y_region = label_padded_region.to(device=args.device, dtype=torch.long)

        y_pred_poi, y_pred_time, y_pred_region = seq_model(x, mask, src_key_padding_mask)

        loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
        loss_time = criterion_time(y_pred_time.transpose(1, 2), y_time)
        loss_region = criterion_region(y_pred_region.transpose(1, 2), y_region)
        loss = loss_poi + loss_region + loss_time
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        top20_acc = 0
        mrr = 0

        batch_label_pois = y_poi.detach().cpu().numpy()
        batch_pred_pois = y_pred_poi.detach().cpu().numpy()
        train_batches_loss_list.append(loss.detach().cpu().numpy())

        for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
            label_pois = label_pois[:seq_len]
            pred_pois = pred_pois[:seq_len, :]
            top20_acc += top_k_acc(label_pois, pred_pois, k=20)
            mrr += MRR_metric(label_pois, pred_pois)

        train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
        train_batches_mrr_list.append(mrr / len(batch_label_pois))

    poi_embed_model.eval()
    user_embed_model.eval()
    time_embed_model.eval()
    region_embed_model.eval()
    embed_fuse_model.eval()
    seq_model.eval()

    val_batches_top1_acc_list = []
    val_batches_top5_acc_list = []
    val_batches_top10_acc_list = []
    val_batches_top20_acc_list = []
    val_batches_mrr_list = []
    val_batches_loss_list = []

    for vb_idx, batch in enumerate(val_loader):
        batch_pred_seqs = []
        batch_seq_lens = []
        batch_seq_embeds = []
        batch_seq_labels_poi = []
        batch_seq_labels_time = []
        batch_seq_labels_region = []
        batch_seq_users = []

        poi_embeddings = poi_embed_model(poi_idxs, edge_index)
        region_embeddings = region_embed_model(region_idxs, edge_index1)
        user_embeddings = user_embed_model(user_idxs, torch.mm(user_poi, poi_embeddings))

        for sample in batch:
            traj_id = sample[0]
            user_id = traj_id.split('_')[0]
            user_idx = user_id2idx_dict[user_id]
            input_seq_emb_list = []

            input_seq = [each[0] for each in sample[1]]
            label_seq = [each[0] for each in sample[2]]
            label_seq_region = [poi2region_dic[each] for each in label_seq]
            label_seq_time = [time_id2idx_dict[each[1]] for each in sample[2]]

            input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings, user_embeddings, region_embeddings))
            batch_seq_embeds.append(input_seq_embed)
            batch_seq_lens.append(len(input_seq))
            batch_seq_users.append(user_idx)
            batch_seq_labels_poi.append(torch.LongTensor(label_seq))
            batch_seq_labels_time.append(torch.LongTensor(label_seq_time))
            batch_seq_labels_region.append(torch.LongTensor(label_seq_region))

        batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
        label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
        label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
        label_padded_region = pad_sequence(batch_seq_labels_region, batch_first=True, padding_value=-1)

        mask = (torch.triu(torch.ones(batch_padded.size(1), batch_padded.size(1))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        src_mask = torch.zeros((batch_padded.size(0) * 2, batch_padded.size(1), batch_padded.size(1)))
        for i in range(src_mask.size(0)):
            src_mask[i] = mask

        src_key_padding_mask = torch.zeros((batch_padded.size(0), batch_padded.size(1)), dtype=torch.bool)
        for i in range(batch_padded.size(0)):
            for j in range(batch_seq_lens[i], batch_padded.size(1)):
                src_key_padding_mask[i][j] = True

        x = batch_padded.to(device=args.device, dtype=torch.float)
        y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
        y_time = label_padded_time.to(device=args.device, dtype=torch.long)
        y_region = label_padded_region.to(device=args.device, dtype=torch.long)
        y_pred_poi, y_pred_time, y_pred_region = seq_model(x, mask, src_key_padding_mask)

        loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
        loss_time = criterion_time(y_pred_time.transpose(1, 2), y_time)
        loss_region = criterion_region(y_pred_region.transpose(1, 2), y_region)
        loss = loss_poi  +loss_region + loss_time

        top1_acc = 0
        top5_acc = 0
        top10_acc = 0
        top20_acc = 0
        mrr = 0

        batch_label_pois = y_poi.detach().cpu().numpy()
        batch_pred_pois = y_pred_poi.detach().cpu().numpy()

        for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
            label_pois = label_pois[:seq_len]
            pred_pois = pred_pois[:seq_len, :]
            top1_acc += top_k_acc(label_pois, pred_pois, k=1)
            top5_acc += top_k_acc(label_pois, pred_pois, k=5)
            top10_acc += top_k_acc(label_pois, pred_pois, k=10)
            top20_acc += top_k_acc(label_pois, pred_pois, k=20)
            mrr += MRR_metric(label_pois, pred_pois)

        val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
        val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
        val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
        val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
        val_batches_mrr_list.append(mrr / len(batch_label_pois))
        val_batches_loss_list.append(loss.detach().cpu().numpy())

    epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
    epoch_train_mrr = np.mean(train_batches_mrr_list)
    epoch_train_loss = np.mean(train_batches_loss_list)

    epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
    epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
    epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
    epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
    epoch_val_mrr = np.mean(val_batches_mrr_list)
    epoch_val_loss = np.mean(val_batches_loss_list)

    if(epoch_val_top1_acc + epoch_val_top5_acc + epoch_val_top10_acc + epoch_val_top20_acc + epoch_val_mrr > best):
        best = epoch_val_top1_acc + epoch_val_top5_acc + epoch_val_top10_acc + epoch_val_top20_acc + epoch_val_mrr
        best_epoch = epoch
        best_epoch_val_loss = epoch_val_loss

        test_batches_top1_acc_list = []
        test_batches_top5_acc_list = []
        test_batches_top10_acc_list = []
        test_batches_top20_acc_list = []
        test_batches_mrr_list = []

        test_batches_loss_list = []
        
        for vb_idx, batch in enumerate(test_loader):
            batch_pred_seqs = []
            batch_seq_lens = []
            batch_seq_users = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_region = []

            poi_embeddings = poi_embed_model(poi_idxs, edge_index)
            region_embeddings = region_embed_model(region_idxs, edge_index1)
            user_embeddings = user_embed_model(user_idxs, torch.mm(user_poi, poi_embeddings))

            for sample in batch:
                traj_id = sample[0]
                user_id = traj_id.split('_')[0]
                user_idx = user_id2idx_dict[user_id]
                input_seq_emb_list = []

                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                label_seq_region = [poi2region_dic[each] for each in label_seq]
                label_seq_time = [time_id2idx_dict[each[1]]for each in sample[2]]

                input_seq_embed = torch.stack(input_traj_to_embeddings(sample, poi_embeddings, user_embeddings, region_embeddings))
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_seq_users.append(user_idx)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.LongTensor(label_seq_time))
                batch_seq_labels_region.append(torch.LongTensor(label_seq_region))          

            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_region = pad_sequence(batch_seq_labels_region, batch_first=True, padding_value=-1)

            mask = (torch.triu(torch.ones(batch_padded.size(1), batch_padded.size(1))) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            src_mask = torch.zeros((batch_padded.size(0) * 2, batch_padded.size(1), batch_padded.size(1)))
            for i in range(src_mask.size(0)):
                src_mask[i] = mask

            src_key_padding_mask = torch.zeros((batch_padded.size(0), batch_padded.size(1)), dtype=torch.bool)
            for i in range(batch_padded.size(0)):
                for j in range(batch_seq_lens[i], batch_padded.size(1)):
                    src_key_padding_mask[i][j] = True

            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.long)
            y_region = label_padded_region.to(device=args.device, dtype=torch.long)
            y_pred_poi, y_pred_time, y_pred_region = seq_model(x, mask, src_key_padding_mask)          

            loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
            loss_time = criterion_time(y_pred_time.transpose(1, 2), y_time)
            loss_region = criterion_region(y_pred_region.transpose(1, 2), y_region)
            loss = loss_poi +loss_region + loss_time

            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mrr = 0

            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi.detach().cpu().numpy()
            batch_pred_regions = y_pred_region.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()

            total_batch_pred_pois = adjust_pred_pro(batch_pred_pois, batch_pred_regions, batch_seq_users, batch_pred_times)

            for label_pois, pred_pois, seq_len in zip(batch_label_pois, total_batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]
                pred_pois = pred_pois[:seq_len, :]

                top1_acc += top_k_acc(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc(label_pois, pred_pois, k=20)
                mrr += MRR_metric(label_pois, pred_pois)

            test_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            test_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            test_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            test_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            test_batches_mrr_list.append(mrr / len(batch_label_pois))

            test_batches_loss_list.append(loss.detach().cpu().numpy())

        epoch_test_loss = np.mean(test_batches_loss_list)

        best_epoch_test_loss = epoch_test_loss

        best_epoch_test_top1_acc = np.mean(test_batches_top1_acc_list)
        best_epoch_test_top5_acc = np.mean(test_batches_top5_acc_list)
        best_epoch_test_top10_acc = np.mean(test_batches_top10_acc_list)
        best_epoch_test_top20_acc = np.mean(test_batches_top20_acc_list)
        best_epoch_test_mrr = np.mean(test_batches_mrr_list)

    print("train_top20_acc:"+str(epoch_train_top20_acc))
    print("train_mrr:"+str(epoch_train_mrr))
    print("epoch_train_loss:"+str(epoch_train_loss))

    print("val_top20_acc:"+str(epoch_val_top20_acc))
    print("val_mrr:"+str(epoch_val_mrr))
    print("epoch_val_loss:"+str(epoch_val_loss))

    print("best_epoch:"+str(best_epoch)+" "+"best_epoch_val_loss:"+str(best_epoch_val_loss)+" "+"best_epoch_test_loss:"+str(best_epoch_test_loss))

    print("total_top1_acc:"+str(best_epoch_test_top1_acc)+" "+"total_top5_acc:"+str(best_epoch_test_top5_acc)+" "+
          "total_top10_acc:"+str(best_epoch_test_top10_acc)+" "+"total_top20_acc:"+str(best_epoch_test_top20_acc)+" "+
          "total_mrr:"+str(best_epoch_test_mrr))

    monitor_loss = epoch_val_loss
    lr_scheduler.step(monitor_loss)