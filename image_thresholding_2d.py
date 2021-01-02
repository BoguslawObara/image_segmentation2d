''' Image Processing Library'''
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import threshold_otsu
from skimage.filters import threshold_local
from skimage.filters import threshold_mean
from skimage.filters import threshold_minimum
from skimage.filters import threshold_triangle
from skimage.filters import threshold_yen
from skimage.filters import threshold_niblack
from skimage.filters import threshold_sauvola
from skimage.filters import threshold_isodata
from skimage.filters import threshold_li

def display_segm(ims, titles, num_cols=5):
  num_rows = np.math.ceil((len(titles)) / num_cols)
  fig, ax = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)

  ax = ax.ravel()
  for i in range(0, len(titles)):
    ax[i].imshow(ims[i], cmap='gray')
    ax[i].set_title(titles[i])
    ax[i].axis('off')

  for i in range(len(titles), num_rows * num_cols):
    fig.delaxes(ax[i])

  fig.tight_layout()
  plt.show()

if __name__ == '__main__':

  # lists
  ims = []
  titles = []

  # load image
  filename = './im/cell2d.png'
  im = io.imread(filename)
  ims.append(im)
  titles.append('image')

  # gobal thresholding
  imth = im > threshold_mean(im)
  ims.append(imth)
  titles.append('gobal: mean')

  imth = im > threshold_minimum(im)
  ims.append(imth)
  titles.append('gobal: minimum')

  imth = im > threshold_triangle(im)
  ims.append(imth)
  titles.append('gobal: minimum')

  imth = im > threshold_isodata(im)
  ims.append(imth)
  titles.append('gobal: isodata')

  imth = im > threshold_otsu(im)
  ims.append(imth)
  titles.append('gobal: otsu')

  imth = im > threshold_li(im)
  ims.append(imth)
  titles.append('gobal: li')

  imth = im > threshold_yen(im)
  ims.append(imth)
  titles.append('gobal: yen')

  # local thresholding
  window_size = 15

  imth = im > threshold_local(im, block_size=window_size, method='mean')
  ims.append(imth)
  titles.append('local: mean')

  imth = im > threshold_local(im, block_size=window_size, method='median')
  ims.append(imth)
  titles.append('local: median')

  imth = im > threshold_local(im, block_size=window_size, method='gaussian')
  ims.append(imth)
  titles.append('local: gaussian')

  imth = im > threshold_niblack(im, window_size=window_size, k=0.8)
  ims.append(imth)
  titles.append('local: niblack')

  imth = im > threshold_sauvola(im, window_size=window_size)
  ims.append(imth)
  titles.append('local: sauvola')

  # plot
  display_segm(ims, titles)
