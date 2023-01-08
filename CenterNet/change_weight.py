import numpy as np
import torch


def Combine_weights(ir_weights,rgb_weights,alpha,save_path):
    # lambda_ = np.random.beta(alpha, alpha)
    lambda_=alpha
    weights_1 = torch.load(ir_weights)
    # print(weights_1['head.reg_head.3.bias'])
    weights_2 = torch.load(rgb_weights)
    for key,value in weights_2.items():
        if key in weights_1:
            weights_1[key]=torch.tensor(((1-lambda_)*weights_1[key].cpu().numpy()+lambda_ * weights_2[key].cpu().numpy()),device ='cuda:0')
    # print(weights_1['head.reg_head.3.bias'])
    torch.save(weights_1,r"D:\xiangmushiyan\centernet-pytorch-main\new_weights.pth")
    return save_path
save_path=r'D:\xiangmushiyan\centernet-pytorch-main'
ir_weights=r"D:\xiangmushiyan\centernet-pytorch-main\logs\loss_2022_10_17_16_36_01-yolov7-ir\best_epoch_weights.pth"
rgb_weights=r"D:\xiangmushiyan\centernet-pytorch-main\logs\loss_2022_10_18_21_51_40-yolov7-rgb\best_epoch_weights.pth"
alpha=0.99

Combine_weights(ir_weights,rgb_weights,alpha,save_path)