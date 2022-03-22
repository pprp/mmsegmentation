import torch 
from collections import OrderedDict

# from_path = '/HOME/scz0088/run/datasets/weights/model_seresnet50.pth.tar'

# save_path = './work_dirs/seresnet50/model_seresnet5.pth.tar'

# para_dict = torch.load(from_path)['state_dict']

# new_dict = OrderedDict()

# print(para_dict.keys())

# for key, value in para_dict.items():
#     # print(key, key.replace('attention', 'receptive_field_attention'))
#     new_dict[key.replace('se', 'attention')] = value 

# torch.save(new_dict, save_path)

from_path = '/HOME/scz0088/run/datasets/weights/model_r50_cbam.pth'

save_path = './work_dirs/resnet50_cbam/model_r50_cbam.pth'

para_dict = torch.load(from_path)['state_dict']

new_dict = OrderedDict()

print(para_dict.keys())

for key, value in para_dict.items():
    # print(key, key.replace('attention', 'receptive_field_attention'))
    new_dict[key.replace('cbam', 'attention')] = value 

torch.save(new_dict, save_path)