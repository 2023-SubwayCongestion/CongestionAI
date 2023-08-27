import cv2
import numpy as np
from scipy.ndimage import label
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM

from mcnn_model import MCNN
from my_dataloader import CrowdDataset

def infer_and_save(model, image_path, save_path):
    '''
    Infer the crowd count and save the overlaid image.

    model: trained MCNN model
    image_path: path to the input image
    save_path: path to save the overlaid image
    '''

    device = torch.device("cuda")

    # Load image
    img = cv2.imread(image_path)
    img_transformed = dataset.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Assuming the dataset has a transform method
    img_transformed = img_transformed.unsqueeze(0).to(device)

    # Infer density map
    model.eval()
    with torch.no_grad():
        et_dmap = model(img_transformed).squeeze(0).squeeze(0).cpu().numpy()

    # Count people by integrating the density map
    crowd_count = int(np.sum(et_dmap))

    # Localize points of maximum densities (peaks)
    labeled, num_features = label(et_dmap > (0.2 * et_dmap.max()))  # The threshold can be tuned
    locations = []
    for i in range(num_features):
        locations.append(np.unravel_index(et_dmap.argmax(), et_dmap.shape))
        et_dmap[locations[-1]] = 0

    # Overlay the points on the image
    for loc in locations:
        y, x = loc
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

    # Save the result
    cv2.imwrite(save_path, img)

    print(f"Estimated crowd count: {crowd_count}")
    return crowd_count

if __name__ == "__main__":
    torch.backends.cudnn.enabled = False

    model_param_path = './checkpoints/epoch_63.param'
    mcnn = MCNN().to(torch.device("cuda"))
    mcnn.load_state_dict(torch.load(model_param_path))

    image_path = '/path/to/your/input/image.jpg'
    save_path = '/path/to/your/output/image.jpg'

    infer_and_save(mcnn, image_path, save_path)
