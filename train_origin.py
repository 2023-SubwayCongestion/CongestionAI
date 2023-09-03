import os
import torch
import torch.nn as nn
import visdom
import random
import argparse
from tqdm import tqdm
from mcnn_model import MCNN
from my_dataloader_origin import CrowdDataset
from models import build_model
import torch.nn.functional as F
import numpy as np
import scipy.spatial
import scipy.ndimage
os.environ["CUDA_VISIBLE_DEVICES"]="7"

def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set parameters for training P2PNet', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)  # 원래 3500
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    # dataset parameters
    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='',
                        help='path where the dataset is')

    parser.add_argument('--output_dir', default='./logs_kd',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./weights_kd',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./logs_kd',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 1 epoch')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='the gpu used for training')

    return parser

def tensor_to_img(tensor):
    # Convert tensor from CxHxW to HxWxC
    img_np = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Convert [-1, 1] range to [0, 1] 
    img_np = (img_np + 1) / 2.0

    # If needed, convert to [0, 255] range
    # img_np = (img_np * 255).astype(np.uint8)

    return img_np

def teacher_density_map(img, teacher_points):
    """
    Generate a density map for the given image using the teacher_points.

    img: input image
    teacher_points: a list of pedestrian locations in the format [[col, row], [col, row], ...]

    return:
    density: the density-map corresponding to the given points. It has the same shape as the input image.
    """
    teacher_points = teacher_points.squeeze(0)
    # Convert to numpy array
    teacher_points = teacher_points.cpu().numpy()
    img_shape = [img.shape[0], img.shape[1]]
    
    # Initialize density map
    density = np.zeros(img_shape, dtype=np.float32)
    
    # If there are no points, return the empty density map
    if len(teacher_points) == 0:
        return density
    
    leafsize = 2048
    # Build KDTree for the given points
    tree = scipy.spatial.KDTree(teacher_points.copy(), leafsize=leafsize)
    
    # Query KDTree for distances
    distances, _ = tree.query(teacher_points, k=4)

    for i, pt in enumerate(teacher_points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        
        # Check boundary conditions
        if 0 <= int(pt[1]) < img_shape[0] and 0 <= int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue
        
        # If more than one point, use average of three smallest distances as sigma
        if len(teacher_points) > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            # If only one point, use average of the image dimensions as sigma
            sigma = np.average(np.array(img_shape)) / 2. / 2.

        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')

    return density


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        'P2PNet knowledge distillation training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    torch.backends.cudnn.enabled=False
    vis=visdom.Visdom()
    device=torch.device("cuda")
    mcnn=MCNN().to(device)
    criterion=nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.Adam(mcnn.parameters())
    # optimizer = torch.optim.SGD(mcnn.parameters(), lr=1e-6,
    #                             momentum=0.95)
    
    img_root='./datasets/ShanghaiTech/part_A_final/train_data/images'
    gt_dmap_root='./datasets/ShanghaiTech/part_A_final/train_data/ground-truth'
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)

    test_img_root='./datasets/ShanghaiTech/part_A_final/test_data/images'
    test_gt_dmap_root='./datasets/ShanghaiTech/part_A_final/test_data/ground-truth'
    test_dataset=CrowdDataset(test_img_root,test_gt_dmap_root,4)
    test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)

    # get the P2PNet model
     # teacher p2pnet
    teacher_model, _ = build_model(args, training=True)
    teacher_weights = "./weights/SHTechA.pth"  # 선생님 모델의 가중치 경로
    teacher_checkpoint = torch.load(teacher_weights, map_location='cuda') # 모델 로드
    teacher_model.load_state_dict(teacher_checkpoint['model'])
    teacher_model.to(device)
    # 각 파라미터의 크기 출력
    for name, param in teacher_model.named_parameters():
        print(f'Parameter name: {name}, Size: {param.size()}')
    # distillation parameter, 일단 0.5로 설정 (distillation loss와 student model loss를 어느 비율로 반영할 것인지를 결정)
    alpha = 0.5

    #training phase
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    min_mae=10000
    min_epoch=0
    train_loss_list=[]
    epoch_list=[]
    test_error_list=[]
    for epoch in range(0,2000):

        mcnn.train()
        # teacher model은 학습 X
        teacher_model.eval()
        epoch_loss=0
        
        for i,(img,gt_dmap) in enumerate(tqdm(dataloader)):
            img_np = tensor_to_img(img) # tensor인 img -> numpy 변환
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            
            # Forward propagation through mcnn (student) and p2pnet (teacher)
            et_dmap = mcnn(img)
            with torch.no_grad():  
                teacher_dmap = teacher_model(img)

            # original MSE loss 계산
            original_loss = criterion(et_dmap, gt_dmap)
            
            # distillation loss 계산
            # Create density map from teacher's output
            to_density_map = teacher_density_map(img_np, teacher_dmap['pred_points'])
            teacher_dmap = torch.tensor(to_density_map).unsqueeze(0).unsqueeze(0).to(device)
            et_dmap = F.interpolate(et_dmap, size=(400, 400), mode='bilinear', align_corners=False)
            # print(et_dmap.shape) # torch.Size([1, 1, 100, 100])
            # print(teacher_density_map.shape) # torch.Size([1, 1, 400, 400])
            distill_loss = criterion(et_dmap, teacher_dmap)
            
            # Combine the original loss and distillation loss
            loss = (1 - alpha) * original_loss + alpha * distill_loss
            
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print("epoch:",epoch,"loss:",epoch_loss/len(dataloader))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss/len(dataloader))
        torch.save(mcnn.state_dict(),'./checkpoints/epoch_'+str(epoch)+".param")

        mcnn.eval()
        mae=0
        for i,(img,gt_dmap) in enumerate(tqdm(test_dataloader)):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img)
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap
        if mae/len(test_dataloader)<min_mae:
            min_mae=mae/len(test_dataloader)
            min_epoch=epoch
        test_error_list.append(mae/len(test_dataloader))
        print("epoch:"+str(epoch)+" error:"+str(mae/len(test_dataloader))+" min_mae:"+str(min_mae)+" min_epoch:"+str(min_epoch))
        vis.line(win=1,X=epoch_list, Y=train_loss_list, opts=dict(title='train_loss'))
        vis.line(win=2,X=epoch_list, Y=test_error_list, opts=dict(title='test_error'))
        # show an image
        index=random.randint(0,len(test_dataloader)-1)
        img,gt_dmap=test_dataset[index]
        vis.image(win=3,img=img,opts=dict(title='img'))
        vis.image(win=4,img=gt_dmap/(gt_dmap.max())*255,opts=dict(title='gt_dmap('+str(gt_dmap.sum())+')'))
        img=img.unsqueeze(0).to(device)
        gt_dmap=gt_dmap.unsqueeze(0)
        et_dmap=mcnn(img)
        et_dmap=et_dmap.squeeze(0).detach().cpu().numpy()
        vis.image(win=5,img=et_dmap/(et_dmap.max())*255,opts=dict(title='et_dmap('+str(et_dmap.sum())+')'))
        


    import time
    print(time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))

        