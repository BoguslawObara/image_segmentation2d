''' Image Processing Library'''
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import color
from skimage import img_as_ubyte, img_as_float
from skimage.transform import rescale
from skimage.segmentation import slic
from skimage.future.graph import cut_normalized
from skimage.future.graph import rag_mean_color
from sklearn.feature_extraction.image import img_to_graph
from sklearn.cluster import spectral_clustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


def normalised_cut_clustering(im):

  # [0-1] -> [0,255]
  im = img_as_ubyte(im)

  # region adjacency graph
  im_labels_rag = slic(im, n_segments=500, compactness=30)

  # normalized cut
  g = rag_mean_color(im, im_labels_rag, mode='similarity')
  im_labels_nc = cut_normalized(im_labels_rag, g, num_cuts=3)

  # labeling
  im_labels_rac = color.label2rgb(im_labels_rag, im, kind='avg')
  im_labels_nc = color.label2rgb(im_labels_nc, im, kind='avg')

  return im_labels_rac, im_labels_nc

def spectral_graph_clustering(im, scale=0.1):

  # resize it to 10% of the original size to speed up the processing
  im_s = rescale(im, scale, anti_aliasing=False)

  # convert the image into a graph with the value of the gradient on the edges
  graph = img_to_graph(im_s)

  # define decreasing function of the gradient
  beta = 10
  eps = 1e-6
  graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

  # clustering
  im_labels = spectral_clustering(graph, n_clusters=2, assign_labels='discretize', random_state=1)

  # resize back
  im_labels = im_labels.reshape(im_s.shape)

  return im_labels

def spectral_graph_clustering_new(im, scale=0.1):

  # resize it to 10% of the original size to speed up the processing
  im_s = rescale(im, scale, anti_aliasing=False)

  # convert the image into a graph with the value of the gradient on the edges
  graph = img_to_graph(im_s)

  # define decreasing function of the gradient
  beta = 10
  eps = 1e-6
  graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

  # model
  model = SpectralClustering(n_clusters=2,
                             affinity='precomputed',
                             assign_labels='discretize',
                             random_state=1)

  # clustering
  labels = model.fit_predict(graph)
  im_labels_s = labels.reshape(im_s.shape)
  im_labels = im_labels_s.reshape(im_s.shape)

  return im_labels

def spectral_nn_clustering(im, scale=0.1):

  # resize it to 10% of the original size to speed up the processing
  im_s = rescale(im, scale, anti_aliasing=False)

  # reshape
  x, y = im_s.shape
  v = im_s.reshape(x*y, 1)

  # model
  model = SpectralClustering(n_clusters=2,
                             eigen_solver='arpack',
                             affinity='nearest_neighbors')

  # clustering
  labels = model.fit_predict(v)

  # reshape back
  im_labels_s = labels.reshape(im_s.shape)
  im_labels = im_labels_s.reshape(im_s.shape)

  return im_labels

def agglomerative_graph_clustering(im, scale=0.1):

  # resize it to 10% of the original size to speed up the processing
  im_s = rescale(im, scale, anti_aliasing=False)

  # reshape
  x, y = im_s.shape
  v = im_s.reshape(x*y, 1)

  # connectivity matrix for structured Ward
  conn = kneighbors_graph(v, n_neighbors=10, include_self=False)

  # make connectivity symmetric
  conn = 0.5 * (conn + conn.T)

  # model
  model = AgglomerativeClustering(n_clusters=2,
                                  linkage='ward',
                                  connectivity=conn)

  # clustering
  labels = model.fit_predict(v)

  # reshape back
  im_labels_s = labels.reshape(im_s.shape)
  im_labels = im_labels_s.reshape(im_s.shape)

  return im_labels

def birch_clustering(im, scale=0.1):

  # resize it to 10% of the original size to speed up the processing
  im_s = rescale(im, scale, anti_aliasing=False)

  # reshape
  x, y = im_s.shape
  v = im_s.reshape(x*y, 1)

  # model
  model = Birch(n_clusters=2, threshold=0.1)

  # clustering
  labels = model.fit_predict(v)

  # reshape back
  im_labels_s = labels.reshape(im_s.shape)
  im_labels = im_labels_s.reshape(im_s.shape)

  return im_labels

def dbscan_clustering(im, scale=0.1):

  # resize it to 10% of the original size to speed up the processing
  im_s = rescale(im, scale, anti_aliasing=False)

  # reshape
  x, y = im_s.shape
  v = im_s.reshape(x*y, 1)

  # model
  model = DBSCAN(eps=0.05, min_samples=100, metric='euclidean')

  # clustering
  labels = model.fit_predict(v)

  # reshape back
  im_labels_s = labels.reshape(im_s.shape)
  im_labels = im_labels_s.reshape(im_s.shape)

  return im_labels

def display_segm(ims, titles, num_cols=4):
  num_rows = np.math.ceil((len(titles)) / num_cols)
  fig, ax = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)

  ax = ax.ravel()
  for i in range(0, len(titles)):
    if i==0:
      ax[i].imshow(im, cmap='gray')
    else:
      ax[i].imshow(ims[i], cmap='jet')
    ax[i].set_title(titles[i])
    ax[i].axis('off')
    ax[i].autoscale(tight=True)

  for i in range(len(titles), num_rows * num_cols):
    fig.delaxes(ax[i])

  #fig.tight_layout()
  plt.show()

if __name__ == '__main__':

  # lists
  ims = []
  titles = []

  # load image
  filename = './im/cell2d.png'
  im = io.imread(filename)
  im = img_as_float(im)
  im = rescale(im, 0.2, anti_aliasing=False)
  ims.append(im)
  titles.append('image')

  # auto semi-supervised image clustering
  _, im_labels = normalised_cut_clustering(im)
  ims.append(im_labels)
  titles.append('normalised_cut_clustering')

  im_labels = spectral_graph_clustering(im, scale=1)
  ims.append(im_labels)
  titles.append('spectral_graph_clustering')

  im_labels = spectral_graph_clustering_new(im, scale=1)
  ims.append(im_labels)
  titles.append('spectral_graph_clustering_new')

  im_labels = spectral_nn_clustering(im, scale=1)
  ims.append(im_labels)
  titles.append('spectral_nn_clustering')

  im_labels = agglomerative_graph_clustering(im, scale=1)
  ims.append(im_labels)
  titles.append('agglomerative_graph_clustering')

  im_labels = birch_clustering(im, scale=1)
  ims.append(im_labels)
  titles.append('birch_clustering')

  im_labels = dbscan_clustering(im, scale=1)
  ims.append(im_labels)
  titles.append('dbscan_clustering')

  # plot
  display_segm(ims, titles)
