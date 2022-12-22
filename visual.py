import shutup
shutup.please()
import torch
import torchvision.models as models
from torch.nn import functional as F

import numpy as np
from PIL import Image
from generalframeworks.networks.deeplabv3.deeplabv3 import DeepLabv3Plus_with_un
from generalframeworks.augmentation.transform import transform

# ++++++++++++++++++++ Utils +++++++++++++++++++++++++
def create_pascal_label_colormap():
  """Creates a label colormap used in Pascal segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = 255 * np.ones((256, 3), dtype=np.uint8)
  colormap[0] = [0, 0, 0]
  colormap[1] = [128, 0, 0]
  colormap[2] = [0, 128, 0]
  colormap[3] = [128, 128, 0]
  colormap[4] = [0, 0, 128]
  colormap[5] = [128, 0, 128]
  colormap[6] = [0, 128, 128]
  colormap[7] = [128, 128, 128]
  colormap[8] = [64, 0, 0]
  colormap[9] = [192, 0, 0]
  colormap[10] = [64, 128, 0]
  colormap[11] = [192, 128, 0]
  colormap[12] = [64, 0, 128]
  colormap[13] = [192, 0, 128]
  colormap[14] = [64, 128, 128]
  colormap[15] = [192, 128, 128]
  colormap[16] = [0, 64, 0]
  colormap[17] = [128, 64, 0]
  colormap[18] = [0, 192, 0]
  colormap[19] = [128, 192, 0]
  colormap[20] = [0, 64, 128]
  return colormap

def color_map(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]
    return np.uint8(color_mask)


# ++++++++++++++++++++ Pascal VOC Visualisation +++++++++++++++++++++++++
# Initialization
im_size = [513, 513]
root = './dataset/pascal'
num_segments = 21
device = torch.device("cpu")
model = DeepLabv3Plus_with_un(models.resnet101(), num_classes=num_segments).to(device)

# Load checkpoint
checkpoint = torch.load('./best_model.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])

# Switch to eval mode
model.eval()

# Generate color map for visualisation
colormap = create_pascal_label_colormap()

# Visualise image in validation set

# Load images and pre-process
with open(root + '/test_val.txt') as f:
    idx_list = f.read().splitlines()
for id in idx_list:
    print('Image {} start!'.format(id))
    im = Image.open('./dataset/VOCdevkit/VOC2012/JPEGImages/{}.jpg'.format(id))
    im.save('./vis/image/{}.png'.format(id))
    gt_label = Image.open('./dataset/VOCdevkit/VOC2012/SegmentationClassAug/{}.png'.format(id))
    im_tensor, label_tensor = transform(im, gt_label, None, crop_size=im_size, scale_size=(1.0, 1.0), augmentation=False)
    im_w, im_h = im.size

    # Inference
    logits, _, _ = model(im_tensor.unsqueeze(0))
    logits = F.interpolate(logits, size=im_size, mode='bilinear', align_corners=True)
    max_logits, label_prcl = torch.max(torch.softmax(logits, dim=1), dim=1)
    
    # Show the results and save
    gt_blend = Image.blend(im, Image.fromarray(color_map(label_tensor[0].numpy(), colormap)[:im_h, :im_w]), alpha=0.7)
    prcl_blend = Image.blend(im, Image.fromarray(color_map(label_prcl[0].numpy(), colormap)[:im_h, :im_w]), alpha=0.7)
    prcl_blend.save('./vis/prcl/{}.png'.format(id))
