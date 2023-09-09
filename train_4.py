import os
import torch
import torch.nn as nn
import visdom
import random
import argparse
from tqdm import tqdm
from mcnn_model import MCNN
from my_dataloader import CrowdDataset
from models import M_SFANet
import torch.nn.functional as F
import numpy as np
import scipy.spatial
import scipy.ndimage

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--data-dir', default='',
                        help='training data directory')
    parser.add_argument('--save-dir', default='',
                        help='model directory')
    parser.add_argument('--device', default='0', help='assign device')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    torch.backends.cudnn.enabled = False
    vis = visdom.Visdom()
    device = torch.device("cuda")
    mcnn = MCNN().to(device)
    criterion = nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.Adam(mcnn.parameters(), lr=1e-4)

    img_root = "./datasets/ShanghaiTech/part_A_final/train_data/images"
    gt_dmap_root = "./datasets/ShanghaiTech/part_A_final/train_data/ground-truth"
    dataset = CrowdDataset(img_root, gt_dmap_root, 4, dataset_name = 'SHA', is_train = True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    test_img_root = "./datasets/ShanghaiTech/part_A_final/test_data/images"
    test_gt_dmap_root = "./datasets/ShanghaiTech/part_A_final/test_data/ground-truth"
    test_dataset = CrowdDataset(test_img_root, test_gt_dmap_root, 4, dataset_name = 'SHA', is_train = False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # get the model
    # teacher
    teacher_model = M_SFANet.Model()
    teacher_weights = "./weights/best_MSFANet_A.pth"  # 선생님 모델의 가중치 경로
    teacher_model.load_state_dict(torch.load(teacher_weights, device)["model"])
    teacher_model.to(device)

    # distillation parameter, 일단 0.5로 설정 (distillation loss와 student model loss를 어느 비율로 반영할 것인지를 결정)
    alpha = 0.5

    # training phase
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    min_mae = 10000
    min_epoch = 0
    train_loss_list = []
    epoch_list = []
    test_error_list = []
    for epoch in tqdm(range(2000), desc="Processing epochs"):
        mcnn.train()
        # teacher model은 학습 X
        teacher_model.eval()
        epoch_loss = 0

        for i, (img, gt_dmap) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Processing batches",
            leave=False,
        ):# tensor인 img -> numpy 변환
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)

            # Forward propagation through mcnn (student) and p2pnet (teacher)
            et_dmap = mcnn(img)

            teacher_dmap,_ = teacher_model(img)

            # original MSE loss 계산
            original_loss = criterion(et_dmap, gt_dmap)
            
            # print("et_dmap:",et_dmap.shape) # et_dmap: torch.Size([1, 1, 100, 100])
            # print("teacher_dmap:",teacher_dmap.shape)  # teacher_dmap: torch.Size([1, 1, 200, 200])
            teacher_dmap_downsampled = F.interpolate(teacher_dmap, size=(100, 100), mode='bilinear', align_corners=False) # 다운샘플링
            # distillation loss 계산
            distill_loss = criterion(et_dmap, teacher_dmap_downsampled)

            # Combine the original loss and distillation loss
            loss = (1 - alpha) * original_loss + alpha * distill_loss

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print("epoch:",epoch,"loss:",epoch_loss/len(dataloader))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss / len(dataloader))
        torch.save(mcnn.state_dict(), "./checkpoints/epoch_" + str(epoch) + ".param")

        mcnn.eval()
        mae = 0
        for i, (img, gt_dmap) in enumerate(test_dataloader):
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            # forward propagation
            et_dmap = mcnn(img)
            mae += abs(et_dmap.data.sum() - gt_dmap.data.sum()).item()
            del img, gt_dmap, et_dmap
        if mae / len(test_dataloader) < min_mae:
            min_mae = mae / len(test_dataloader)
            min_epoch = epoch
        test_error_list.append(mae / len(test_dataloader))
        print(
            "epoch:"
            + str(epoch)
            + " error:"
            + str(mae / len(test_dataloader))
            + " min_mae:"
            + str(min_mae)
            + " min_epoch:"
            + str(min_epoch)
        )
        vis.line(win=1, X=epoch_list, Y=train_loss_list, opts=dict(title="train_loss"))
        vis.line(win=2, X=epoch_list, Y=test_error_list, opts=dict(title="test_error"))
        # show an image
        index = random.randint(0, len(test_dataloader) - 1)
        img, gt_dmap = test_dataset[index]
        vis.image(win=3, img=img, opts=dict(title="img"))
        vis.image(
            win=4,
            img=gt_dmap / (gt_dmap.max()) * 255,
            opts=dict(title="gt_dmap(" + str(gt_dmap.sum()) + ")"),
        )
        img = img.unsqueeze(0).to(device)
        gt_dmap = gt_dmap.unsqueeze(0)
        et_dmap = mcnn(img)
        et_dmap = et_dmap.squeeze(0).detach().cpu().numpy()
        vis.image(
            win=5,
            img=et_dmap / (et_dmap.max()) * 255,
            opts=dict(title="mcnn_dmap(" + str(et_dmap.sum()) + ")"),
        )

    import time

    print(time.strftime("%Y.%m.%d %H:%M:%S", time.localtime(time.time())))
