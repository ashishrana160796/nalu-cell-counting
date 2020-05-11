import scipy.ndimage
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Changing figure's font to Times New Roman
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"



def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    
    if titles is None: titles = ['Cell image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    plt.title('Data augmentation techniques', fontsize=32)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title, fontsize=22)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

im_list = ['assets/data-aug-images/a.jpeg',
           'assets/data-aug-images/b.jpeg',
           'assets/data-aug-images/c.jpeg',
           'assets/data-aug-images/d.jpeg',
           'assets/data-aug-images/e.jpeg',
           'assets/data-aug-images/f.jpeg',
           'assets/data-aug-images/g.jpeg',
           'assets/data-aug-images/h.jpeg',
           'assets/data-aug-images/i.jpeg',
           'assets/data-aug-images/j.jpeg']


im_a = scipy.ndimage.imread('assets/data-aug-images/a.jpeg', True)
im_b = scipy.ndimage.imread('assets/data-aug-images/b.jpeg', True)
im_c = scipy.ndimage.imread('assets/data-aug-images/c.jpeg', True)
im_d = scipy.ndimage.imread('assets/data-aug-images/d.jpeg', True)
im_e = scipy.ndimage.imread('assets/data-aug-images/e.jpeg', True)
im_f = scipy.ndimage.imread('assets/data-aug-images/f.jpeg', True)
im_g = scipy.ndimage.imread('assets/data-aug-images/g.jpeg', True)
im_h = scipy.ndimage.imread('assets/data-aug-images/h.jpeg', True)
im_i = scipy.ndimage.imread('assets/data-aug-images/i.jpeg', True)
im_j = scipy.ndimage.imread('assets/data-aug-images/j.jpeg', True)

im_list = []
im_list.append(im_a)
im_list.append(im_b)
im_list.append(im_c)
im_list.append(im_d)
im_list.append(im_e)
im_list.append(im_f)
im_list.append(im_g)
im_list.append(im_h)
im_list.append(im_i)
im_list.append(im_j)

show_images(im_list, 2, None)






