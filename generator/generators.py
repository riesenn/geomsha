import numpy as np


def shapes(generator=np.random.default_rng(100), num_shapes=25,max_radius=1/20,min_radius=1/40):
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
