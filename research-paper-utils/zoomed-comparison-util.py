import scipy
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt 
# Reference Solution Given By: https://stackoverflow.com/users/1461210/ali-m

# Changing figure's font to Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

def clipped_zoom(img, zoom_factor, **kwargs):
    # assumption is that zoom factor is greater than 1
    h, w = img.shape[:2]
    # no RGB zoom to be applied for multi-channel image,  
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    # Bounding box of the zoomed-in region within the input array
    zh = int(np.round(h / zoom_factor))
    zw = int(np.round(w / zoom_factor))
    top = (h - zh) // 2
    left = (w - zw) // 2
    out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
    # trim off any extra pixels at the edges from 'out'
    trim_top = ((out.shape[0] - h) // 2)
    trim_left = ((out.shape[1] - w) // 2)
    out = out[trim_top:trim_top+h, trim_left:trim_left+w]
    return out
    
# Reading, Zooming, Plotting & Saving Images
# matplotlib.pyplot.imread can also be used as imread is deprecated

im_org = scipy.ndimage.imread('assets/001cell.png', True)
zm_org = clipped_zoom(im_org, 2.0)
im_ano = scipy.ndimage.imread('assets/001dots.png', True)
zm_ano = clipped_zoom(im_ano, 2.0)
fig, ax = plt.subplots(2, 2)

ax[0][0].imshow(im_org)
ax[0][1].imshow(zm_org)
ax[1][0].imshow(im_ano)
ax[1][1].imshow(zm_ano)

ax[0][0].set_title('(a) Normal cell image')
ax[0][1].set_title('(b) Zoomed cell image')
ax[1][0].set_title('(c) Normal annotated image')
ax[1][1].set_title('(d) Zoomed annotated image')
plt.tight_layout(pad=0.2)
plt.savefig('final_assets/cells-fig-compared.png')
