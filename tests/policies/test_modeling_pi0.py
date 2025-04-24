from lerobot.common.policies.pi0.modeling_pi0 import resize_with_pad
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch

img_path_base = "/home/cuhk/"
# need b,c,h,w
img1 = np.array(Image.open(img_path_base+"krotex_table.jpg")).transpose(2, 0, 1)
img2 = np.array(Image.open(img_path_base+"krotex_table2.jpg")).transpose(2, 0, 1)
img_list = torch.from_numpy(np.array([img1, img2]))
img_new = resize_with_pad(img_list, 224, 224, 0)

ax_list = [plt.subplot(1, len(img_new), i+1) for i in range(len(img_new))]
for idx in range(len(img_new)):
    ax_list[idx].imshow(img_new[idx].numpy().transpose(1, 2, 0))
plt.suptitle('test resize with pad')
plt.pause(30)
plt.close()