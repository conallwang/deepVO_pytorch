import torch

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path of all sequences
dataset_path = '/home/share/dataset/sequences'

# save
model_path = '/home/wangcong/workspace/ICRA 2021/DeepVO/checkpoint'
log_path = '/home/wangcong/workspace/ICRA 2021/DeepVO/log'

# path of gt poses
poses_path = '/home/wangcong/workspace/ICRA 2021/DeepVO/poses'

# loss weight
kapa = 100

# seq
train_seq = [0, 1, 2, 8, 9]
test_seq = [3, 4, 5, 6, 7, 10]
vaild_seq = [5]