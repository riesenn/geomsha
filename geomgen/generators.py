import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

def gen_shapes(generator=np.random.default_rng(100), num_shapes=25,max_radius=1/20,min_radius=1/40):
  params = np.zeros((num_shapes,5))
  params[:,0] = generator.integers(2,5,size=num_shapes)
  params[:,1:5] = generator.random((num_shapes,4))
  for p in params:
    p[3] = min_radius+p[3]*(max_radius-min_radius)
    p[1:3] = p[1:3]*(1-2*p[3])+p[3]
    if p[0] < 3:
      p[4] = 0
    else:
      p[4] = p[4]*2*np.pi/p[0]
  return params

def gen_noise(generator=np.random.default_rng(100), num_noise=700, max_line=1/30):
  params = generator.random((num_noise,4))
  params[:,0:2] = params[:,0:2]*(1-2*max_line)+max_line
  params[:,2] = params[:,2]*max_line
  params[:,3] = params[:,3]*2*np.pi
  return params


def gen_image(shapes, noise = None, im_size=160, lw_shape=0.15, lw_noise=0.1):
  sha = shapes.copy()
  sha[:,1:4] = sha[:,1:4]*im_size
  fig, ax = plt.subplots(figsize=(1,1), dpi=im_size)
  plt.axis('scaled')
  plt.axis('off')
  plt.xlim(0, im_size)
  plt.ylim(0, im_size)
  plt.subplots_adjust(bottom=0.0, left=0.0, right=1.0, top=1.0)
  for s in sha:
    if s[0] < 3:
      ax.add_patch(matplotlib.patches.Circle(s[1:3], radius=s[3], lw=lw_shape, fc='b', fill=False))
    else:
      ax.add_patch(matplotlib.patches.RegularPolygon(s[1:3],numVertices=int(s[0]),radius=s[3],orientation=s[4],lw=lw_shape,fc='b',fill=False))
  if not noise is None:
    nse = noise.copy()
    nse[:,0:3] = nse[:,0:3]*im_size
    for n in nse:
      ax.add_line(matplotlib.lines.Line2D((n[0],n[0]+[np.cos(n[3])*n[2]]),(n[1],n[1]+[np.sin(n[3])*n[2]]),lw=lw_noise,c='k'))
  else:
    nse = None  
  fig.set(figwidth=1, figheight=1, dpi=im_size)
  fig.canvas.draw()
  img = np.array(fig.canvas.renderer.buffer_rgba())
  return (img[:,:,0],sha,nse) 
