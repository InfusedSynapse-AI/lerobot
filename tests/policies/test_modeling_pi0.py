from lerobot.common.policies.pi0.modeling_pi0 import resize_with_pad
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img_path_base = "/home/cuhk/quebinbin/workspace/projects/"
img1 = np.array(Image.open(img_path_base+"cam_exterior.png")).transpose(2, 0, 1)
img2 = np.array(Image.open(img_path_base+"cam_wrist.png")).transpose(2, 0, 1)
img_list = np.array([img1, img2])
img_new = resize_with_pad(img_list, 224, 224, 0)

ax_list = [plt.subplot(1, len(img_new), i+1) for i in range(len(img_new))]
for idx in range(len(img_new)):
    ax_list[idx].imshow(img_new[idx].transpose(2, 0, 1))
plt.suptitle('test resize with pad')
plt.pause(10)
plt.close()