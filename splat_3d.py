import argparse
import math
from pathlib import Path
from typing import Tuple
import open3d as o3d
from scan_tools.scan_roi import scan_roi

import numpy as np
from camera_geometry import FrameSet
from geometry_grid.taichi_geometry.conversion import struct_size

import torch
import taichi as ti
from taichi.math import vec3, vec4, mat3

from transform import join_qt, quat_to_mat



@ti.data_oriented
class AABox:
  lower:ti.math.vec3
  upper:ti.math.vec3

  @ti.func
  def extents(self) -> ti.math.vec3:
    return self.upper - self.lower

@ti.func 
def splat_bounds(p:vec3, q:vec4, scale:vec3) -> AABox:
  axes = scale * quat_to_mat(q)

  lower, upper = ti.math.vec3(np.inf), ti.math.vec3(-np.inf)
  for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
    corner = axes @ ti.math.vec3(i, j, k)
    lower = ti.min(lower, corner)
    upper = ti.max(upper, corner)

  return AABox(lower + p, upper + p)

@ti.func 
def to_covariance(q:ti.math.vec4, scale:ti.math.vec3) -> ti.math.mat3:
  axes = scale * quat_to_mat(q)
  return axes @ axes.T

@ti.func
def gaussian(x:ti.math.vec3, mean:ti.math.vec3, inv_cov:ti.math.mat3) -> ti.f32:
  x = x - mean
  return ti.exp(-0.5 * ti.math.dot(x, inv_cov @ x))



@ti.dataclass
class Splat3D:
  p: vec3
  q: vec4
  scale: vec3
  color: vec3
  opacity: ti.f32

  @ti.func
  def rotation(self):
    return quat_to_mat(self.q)


  @ti.func
  def from_vec(self, v:ti.template()):
    self.p = v[0:3]
    self.q = v[3:7]
    self.scale = v[7:10]
    self.color = v[10:13]
    self.opacity = v[13]
  
spat3d_vec=ti.types.vector(dtype=ti.f32, n=struct_size(Splat3D))


