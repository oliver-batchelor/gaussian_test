from splat_3d import Splat3D, spat3d_vec
import numpy as np

import taichi as ti
import taichi.math as tm

def random_rotation():
  # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
  theta = np.random.uniform(0, 2 * np.pi)
  u = np.random.uniform(-1, 1, size=3)
  u /= np.linalg.norm(u)
  w = np.cos(theta / 2)
  x, y, z = u * np.sin(theta / 2)
  return np.array([w, x, y, z])


def random_splats(n=10):
  splats = Splat3D.field(shape=n)

  for i in range(n):
    splat = Splat3D(
      p = tm.vec3(np.random.uniform(-2, 2, size=3)),
      q = tm.vec4(random_rotation()),
      scale = tm.vec3(np.random.uniform(0.01, 0.1, size=3)),
      color = tm.vec3(np.random.uniform(0, 1, size=3)),
      opacity = 1.0)

    splats[i] = splat

  return splats