import torch
import taichi as ti
from taichi.math import vec3, vec4, mat3
import taichi.math as tm


@ti.func
def quat_to_mat(q:vec4) -> mat3:
  w, x, y, z = q
  x2, y2, z2 = x*x, y*y, z*z

  return mat3(
    1 - 2*y2 - 2*z2, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
    2*x*y + 2*w*z, 1 - 2*x2 - 2*z2, 2*y*z - 2*w*x,
    2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x2 - 2*y2
  )

@ti.func
def mat_to_quat(m:mat3) -> vec4:

  m00, m01, m02 = m[0]
  m10, m11, m12 = m[1]
  m20, m21, m22 = m[2]

  t = m00 + m11 + m22

  if t > 0:
    s = 0.5 / ti.sqrt(t + 1)
    w = 0.25 / s
    x = (m21 - m12) * s
    y = (m02 - m20) * s
    z = (m10 - m01) * s
  elif m00 > m11 and m00 > m22:
    s = 2 * ti.sqrt(1 + m00 - m11 - m22)
    w = (m21 - m12) / s
    x = 0.25 * s
    y = (m01 + m10) / s
    z = (m02 + m20) / s
  elif m11 > m22:
    s = 2 * ti.sqrt(1 + m11 - m00 - m22)
    w = (m02 - m20) / s
    x = (m01 + m10) / s
    y = 0.25 * s
    z = (m12 + m21) / s
  else:
    s = 2 * ti.sqrt(1 + m22 - m00 - m11)
    w = (m10 - m01) / s
    x = (m02 + m20) / s
    y = (m12 + m21) / s
    z = 0.25 * s

  return vec4(w, x, y, z)

@ti.func
def scaling(scale:vec3) -> mat3:
  return mat3(
    scale[0], 0, 0,
    0, scale[1], 0,
    0, 0, scale[2]
  )


@ti.func
def join_rt(r:tm.mat3, t:tm.vec3) -> tm.mat4:
  m = tm.mat4(0)
  m[0:3, 0:3] = r
  m[0:3, 3] = t
  m[3, 3] = 1.0
  return m


@ti.func
def join_qt(q:tm.vec4, t:tm.vec3) -> tm.mat4:
  return join_rt(quat_to_mat(q), t)
