import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


def gen_shapes(rnd=np.random.default_rng(), num_shapes=25,max_radius=1/20,min_radius=1/40, no_rotation=False, no_scaling=False):
  params = np.zeros((num_shapes,5))
  params[:,0] = rnd.integers(2,5,size=num_shapes)
  params[:,1:5] = rnd.random((num_shapes,4))
  if no_rotation: params[:,4] = 0
  if no_scaling: params[:,3] = max_radius
  else: params[:,3] = min_radius+params[:,3]*(max_radius-min_radius)
  params[:,1] = params[:,3]+np.multiply(params[:,1],1-2*params[:,3])
  params[:,2] = params[:,3]+np.multiply(params[:,2],1-2*params[:,3])
  idx = params[:,0] < 3
  params[idx,4] = 0
  idx = np.logical_not(idx)
  params[idx,4] = np.multiply(params[idx,4],2*np.pi/params[idx,0])
  return params


def gen_noise(rnd=np.random.default_rng(), num_noise=500, max_line=1/20, min_line=1/80):
  params = rnd.random((num_noise,4))
  params[:,2] = min_line+params[:,2]*(max_line-min_line)
  params[:,0] = params[:,2]+np.multiply(params[:,0],1-2*params[:,2])
  params[:,1] = params[:,2]+np.multiply(params[:,1],1-2*params[:,2])
  params[:,2:4] = np.c_[params[:,0]+np.multiply(params[:,2],np.cos(params[:,3]*2*np.pi)),params[:,1]+np.multiply(params[:,2],np.sin(params[:,3]*2*np.pi))]
  return params


def gen_image(shapes, noise = None, rnd=np.random.default_rng(), im_size=160, max_lw=0.15, min_lw=0.1, show_center=False):
  sha = np.c_[shapes.copy(),rnd.random(len(shapes))]
  sha[:,1:4] = sha[:,1:4]*im_size
  sha[:,5] = min_lw+sha[:,5]*(max_lw-min_lw)
  fig, ax = plt.subplots(figsize=(1,1), dpi=im_size)
  plt.axis('scaled')
  plt.axis('off')
  plt.xlim(0, im_size)
  plt.ylim(0, im_size)
  plt.subplots_adjust(bottom=0.0, left=0.0, right=1.0, top=1.0)
  for s in sha:
    if s[0] < 3:
      ax.add_patch(matplotlib.patches.Circle(s[1:3], radius=s[3], lw=s[5], fc='b', fill=False))
      if show_center: ax.add_patch(matplotlib.patches.Circle(s[1:3], radius=.5, lw=2, fc='b'))
    else:
      ax.add_patch(matplotlib.patches.RegularPolygon(s[1:3],numVertices=int(s[0]),radius=s[3],orientation=s[4],lw=s[5],fc='b',fill=False))
      if show_center: ax.add_patch(matplotlib.patches.Circle(s[1:3], radius=.5, lw=2, fc='b'))
  if noise is None: nse = None
  else:
    nse = np.c_[noise.copy(),rnd.random(len(noise))]
    nse[:,0:4] = nse[:,0:4]*im_size
    nse[:,4] = min_lw+nse[:,4]*(max_lw-min_lw)
    for n in nse:
      ax.add_line(matplotlib.lines.Line2D((n[0],n[2]),(n[1],n[3]),lw=n[4],c='k'))
  fig.set(figwidth=1, figheight=1, dpi=im_size)
  fig.canvas.draw()
  img = np.array(fig.canvas.renderer.buffer_rgba())
  return (img[:,:,0],sha,nse) 
