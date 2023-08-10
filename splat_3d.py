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
from taichi.math import vec3, vec4, mat3, mat4

from transform import join_qt, quat_to_mat, scaling



@ti.data_oriented
class AABox:
  lower:vec3
  upper:vec3

  @ti.func
  def extents(self) -> vec3:
    return self.upper - self.lower

@ti.func 
def splat_bounds(p:vec3, q:vec4, scale:vec3) -> AABox:
  axes = scale * quat_to_mat(q)

  lower, upper = vec3(np.inf), vec3(-np.inf)
  for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
    corner = axes @ vec3(i, j, k)
    lower = ti.min(lower, corner)
    upper = ti.max(upper, corner)

  return AABox(lower + p, upper + p)

@ti.func 
def to_covariance(q:ti.math.vec4, scale:vec3) -> ti.math.mat3:
  axes = scale * quat_to_mat(q)
  return axes @ axes.T

@ti.func
def eval_gaussian(x:vec3, mean:vec3, inv_cov:ti.math.mat3) -> ti.f32:
  v = x - mean
  return ti.exp(-0.5 * ti.math.dot(v, inv_cov @ v))





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




# @ti.func
# def intersect_ray(dir : vec3)


@ti.dataclass
class Gaussian3D:
  mean : vec3
  inv_sigma : mat3
  color : vec3

@ti.func
def to_gaussian_3d(splat : Splat3D, camera_T_world: mat4) -> Gaussian3D:
  scale = scaling(splat.scale)
  axes = (camera_T_world[:3, :3] @ (quat_to_mat(splat.q) @ scale))

  # J @ axes @ axes^T @ J^T
  cov_3d = axes @ axes.transpose()
  print(cov_3d)
  print(cov_3d.inverse())

  return Gaussian3D(
    mean = (camera_T_world @ vec4(splat.p, 1.0)).xyz,
    inv_sigma = cov_3d.inverse(),
    color = splat.color
  )

@ti.kernel
def to_gaussians_3d(splats : ti.template(), camera_T_world: mat4, gaussians : ti.template()):
  for i in range(splats.shape[0]):
    gaussians[i] = to_gaussian_3d(splats[i], camera_T_world)

@ti.func
def ray_gaussian_3d(d : vec3, g : Gaussian3D) -> ti.f32:
  x = g.mean.dot(g.inv_sigma @ d)  + d.dot(g.inv_sigma @ g.mean)  
  denom = 2 * d.dot(g.inv_sigma @ d)
  return x / denom    