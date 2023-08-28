import cv2
import numpy as np
from scipy.ndimage import label
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from PIL import Image
from torchvision import transforms as T
from mcnn_model import MCNN
from my_dataloader import CrowdDataset

def inferenceMCNN(model, image_path, save_path):
    '''
    Infer the crowd count and save the overlaid image.

    model: trained MCNN model
    image_path: path to the input image
    save_path: path to save the overlaid image
    '''

    device = torch.device("cuda")

    # 이미지 로드
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)  # PIL 이미지를 Numpy array로 변환
    #image = image.convert('RGB')
    
    # 이미지 전처리
    transform = T.Compose([
        T.Resize((400, 400)),  # 예: 원래 크기에 맞게 조정해야 합니다.
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(image)
    img = img.unsqueeze(0)  # 배치 차원 추가
    img = img.to(device)

    # density map 추론
    model.eval()
    et_dmap = model(img)

    # density map 더하여 총 인원 수 카운트
    crowd_count = int(torch.sum(et_dmap).item())

    # Localize points of maximum densities (peaks)
    et_dmap_cpu = et_dmap.detach().cpu().numpy()
    # Normalize
    dmap_norm = (et_dmap_cpu[0,0] - et_dmap_cpu[0,0].min()) / (et_dmap_cpu[0,0].max() - et_dmap_cpu[0,0].min())
    dmap_scaled = (255 * dmap_norm).astype(np.uint8)

    # Save density map to image
    cv2.imwrite(save_path, dmap_scaled)
    # labeled, num_features = label(et_dmap_cpu > (0.5 * et_dmap_cpu.max()))  # threshold 0.5

    # locations = []
    # for i in range(num_features):
    #     max_val_idx = et_dmap.argmax().item()
    #     loc = np.unravel_index(max_val_idx, et_dmap[0, 0].shape)  # 2D shape를 사용
    #     locations.append(loc)
    #     et_dmap[0, 0, loc[0], loc[1]] = 0
    # print(locations)
    # # 이미지 위에 결과 중첩
    # for loc in locations:
    #     x, y = loc
    #     cv2.circle(img_array, (x, y), 5, (0, 0, 255), -1)

    # # 결과 저장
    # cv2.imwrite(save_path, img_array)

    print(f"Estimated crowd count: {crowd_count}")
    return crowd_count

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False

    model_param_path = './checkpoints/epoch_test.param'
    mcnn = MCNN().to(torch.device("cuda"))
    mcnn.load_state_dict(torch.load(model_param_path))

    image_path = './demo/image.png'
    save_path = './demo/result/image.png'

    inferenceMCNN(mcnn, image_path, save_path)
