''' Image Processing Library'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.segmentation import random_walker

class Cursor(object):
  def __init__(self, fig, ax1, ax2, ax3, im, im_markers, im_labels, r):
    self.fig = fig
    self.ax1 = ax1
    self.ax2 = ax2
    self.ax3 = ax3
    self.im = im
    self.im_markers = im_markers
    self.im_labels = im_labels
    self.r = r
    self.button = 0
    self.xs = self.im.shape[0]
    self.ys = self.im.shape[1]
    self.xr = np.linspace(0, self.xs-1, self.xs)
    self.yr = np.linspace(0, self.ys-1, self.ys)
    self.yy, self.xx = np.meshgrid(self.yr, self.xr)

    self.ax1.imshow(self.im, cmap='gray', interpolation='nearest')
    self.ax1.axis('off')
    self.ax1.set_title('Noisy data')
    self.ax2.imshow(self.im_markers, cmap='magma', interpolation='nearest')
    self.ax2.axis('off')
    self.ax2.set_title('Markers')
    self.ax3.imshow(self.im_labels, cmap='gray', interpolation='nearest')
    self.ax3.axis('off')
    self.ax3.set_title('Segmentation')
    self.fig.tight_layout()

  def draw(self, event):
    x, y = event.xdata, event.ydata
    x, y = int(x), int(y)
    # print('x=%1.2f, y=%1.2f' % (x, y))
    #imd = np.ones(im.shape, dtype=np.bool)
    #imd[y, x] = 0
    #imd = distance_transform_edt(imd)<self.r
    imd = np.sqrt((self.xx-y)**2 + (self.yy-x)**2) <= self.r

    if event.button == 1:
      self.im_markers[imd] = 1
    elif event.button == 3:
      self.im_markers[imd] = 2
    elif event.button == 2:
      self.im_labels = random_walker(self.im, self.im_markers, beta=10, mode='cg_mg') # 'bf'
      self.ax3.imshow(self.im_labels, cmap='gray', interpolation='nearest')
    self.ax2.imshow(self.im_markers, cmap='magma', interpolation='nearest')
    plt.draw()

  def mouse_move(self, event):
    if not event.inaxes:
      return

    if self.button == 1:
      self.draw(event)

  def button_press(self, event):
    if not event.inaxes:
      return

    self.button = 1
    self.draw(event)

  def button_release(self, event):
    if not event.inaxes:
      return
    self.button = 0

if __name__ == '__main__':

  # load image
  filename = './im/cell2d.png'
  im = mpimg.imread(filename)
  im = rgb2gray(im) # intensity = [0,1]

  # manual semi-supervised image clustering
  im_markers = np.zeros(im.shape, dtype=np.uint8)
  im_labels = np.zeros(im.shape, dtype=np.bool)
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3.2),
                                      sharex=False, sharey=False)
  cursor = Cursor(fig, ax1, ax2, ax3, im, im_markers, im_labels, 10)
  plt.connect('motion_notify_event', cursor.mouse_move)
  plt.connect('button_press_event', cursor.button_press)
  plt.show()
  