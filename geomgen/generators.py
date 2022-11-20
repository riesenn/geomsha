import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


def gen_shapes(rnd=np.random.default_rng(100), num_shapes=25,max_radius=1/20,min_radius=1/40):
  params = np.zeros((num_shapes,5))
  params[:,0] = rnd.integers(2,5,size=num_shapes)
  params[:,1:5] = rnd.random((num_shapes,4))
  params[:,3] = min_radius+params[:,3]*(max_radius-min_radius)
  for p in params:
    p[1:3] = p[1:3]*(1-2*p[3])+p[3]
    if p[0] < 3:
      p[4] = 0
    else:
      p[4] = p[4]*2*np.pi/p[0]
  return params


def gen_noise(rnd=np.random.default_rng(100), num_noise=500, max_line=1/20, min_line=1/80):
  params = rnd.random((num_noise,4))
  params[:,2] = min_line+params[:,2]*(max_line-min_line)
  params[:,3] = params[:,3]*2*np.pi
  for p in params:
    p[0:2] = p[0:2]*(1-2*p[2])+p[2]
  return params


def gen_image(shapes, noise = None, rnd=np.random.default_rng(100), im_size=160, max_lw=0.15, min_lw=0.1):
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
    else:
      ax.add_patch(matplotlib.patches.RegularPolygon(s[1:3],numVertices=int(s[0]),radius=s[3],orientation=s[4],lw=s[5],fc='b',fill=False))
  if not noise is None:
    nse = np.c_[noise.copy(),rnd.random(len(noise))]
    nse[:,0:3] = nse[:,0:3]*im_size
    nse[:,4] = min_lw+nse[:,4]*(max_lw-min_lw)
    for n in nse:
      ax.add_line(matplotlib.lines.Line2D((n[0],n[0]+[np.cos(n[3])*n[2]]),(n[1],n[1]+[np.sin(n[3])*n[2]]),lw=n[4],c='k'))
  else:
    nse = None  
  fig.set(figwidth=1, figheight=1, dpi=im_size)
  fig.canvas.draw()
  img = np.array(fig.canvas.renderer.buffer_rgba())
  return (img[:,:,0],sha,nse) 
