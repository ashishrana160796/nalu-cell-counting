import scipy
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt 

# Changing figure's font to Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

im_org = scipy.ndimage.imread('assets/10001cell.png', True)
im_cust = scipy.ndimage.imread('assets/16-10001cell.png', True)

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(im_org)
ax.set_title('Normal cell image', fontsize=24)

ax = fig.add_subplot(1, 2, 2)
ax.set_title('High count cell image', fontsize=24)
ax.imshow(im_cust)

plt.tight_layout(pad=1.0)
plt.savefig('final_assets/bbbc005cells-fig-compared.png')
