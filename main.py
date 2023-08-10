from dataclasses import dataclass
import math
from typing import Tuple
import taichi as ti
import taichi.math as tm
import numpy as np

import open3d as o3d
from camera import Camera

from rand import random_splats
from scipy.spatial.transform import Rotation as R
from splat_2d import Splat2D, density_from_conic, project_splat
from splat_3d import Gaussian3D, eval_gaussian, ray_gaussian_3d, to_gaussian_3d, to_gaussians_3d

from transform import quat_to_mat, scaling


    
def sphere_mesh(n, radius=1.0):
  sphere =  o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=n)

  vertices = np.asarray(sphere.vertices)
  indices = np.asarray(sphere.triangles).flatten()

  vertices_field = ti.field(dtype=tm.vec3, shape=vertices.shape[0])
  indices_field = ti.field(dtype=ti.int32, shape=indices.shape[0])

  vertices_field.from_numpy(vertices)
  indices_field.from_numpy(indices)

  return vertices_field, indices_field


@dataclass
class Options:
   show_splats: bool = False
   trace_splats: bool = False


def show_options(window, options):

    window.GUI.begin("Display Panel", 0.05, 0.1, 0.2, 0.15)
    # display_mode = window.GUI.slider_int("Value Range", some_int_type_value, 0, 5)
    
    options.show_splats = window.GUI.checkbox("Show Splats", options.show_splats)
    options.trace_splats = window.GUI.checkbox("Trace Splats", options.trace_splats)

    
    
    window.GUI.end()



@ti.kernel
def splat_transforms(splats:ti.template(), transforms:ti.template()):
  for i in range(splats.shape[0]):
    splat = splats[i]
    rotation = quat_to_mat(splat.q)

    m = tm.mat4(0)
    m[0:3, 0:3] = rotation @ scaling(splat.scale * 3)
    m[0:3, 3] = splat.p
    m[3, 3] = 1.0

    transforms[i] = m

    
def intrinsic_matrix(hfov_deg, window):
    width, height = window.get_window_shape()
    hfov = math.radians(hfov_deg)

    fx = 0.5 * width / np.tan(hfov * 0.5)


    cx = width * 0.5
    cy = height * 0.5

    return np.array([
        [fx, 0, cx],
        [0, fx, cy],
        [0, 0, 1.0]
    ])
   
@ti.kernel
def draw_splats(splats:ti.template(), image:ti.template(), 
                intrinsic:tm.mat3, view:tm.mat4, near:ti.f32):
  camera = Camera(image.shape[:2], intrinsic)
  w, h = image.shape[:2]
  upper = tm.ivec2(w - 1, h - 1) 

  # for p in ti.grouped(ti.ndrange(image.shape[0], image.shape[1])):
  #   image[p] = 0.0

  for i in splats:
    splat2d = project_splat(splats[i], camera, view)
    if splat2d.depth < near:
      continue

    lower = ti.math.clamp(ti.floor(splat2d.uv - splat2d.radius, ti.i32), 0, upper)
    upper = ti.math.clamp(ti.ceil(splat2d.uv + splat2d.radius, ti.i32), 0, upper)
    
    for x, y in ti.ndrange((lower.x, upper.x), (lower.y, upper.y)):
      xy = ti.math.vec2(x, y)
      density = density_from_conic(xy, splat2d.uv, splat2d.conic)

      density = ti.cast(density > 0.01, ti.f32)
      image[x, y] = tm.mix(image[x, y], splat2d.color, density)# density * splat2d.opacity)



@ti.kernel
def trace_splats(gaussians:ti.template(), image:ti.template(), 
                intrinsic:tm.mat3, near:ti.f32):
  camera = Camera(image.shape[:2], intrinsic)
  w, h = image.shape[:2]

  unproj = camera.image_t_camera.inverse()
  
  for x, y in ti.ndrange(w, h):
    for i in range(gaussians.shape[0]):
      pixel = tm.vec3(x, y, 1.0)
      d = tm.normalize(unproj @ pixel)

      g = gaussians[i]
      t = ray_gaussian_3d(d, g)

      if t > near:
        density = eval_gaussian(d * t, g.mean, g.inv_sigma)
        image[x, y] = tm.mix(image[x, y], g.color, density)


      # density = ti.cast(density > 0.01, ti.f32)
      # image[p] = tm.mix(image[p], splat2d.color, density)

@ti.kernel
def make_checkerboard(image:ti.template(), size:ti.template()):
  for i, j in ti.ndrange(image.shape[0], image.shape[1]):
    image[i, j] = ((i // size + j // size) % 2) * 0.25

def diag_matrix(v):
  m = tm.mat4(0)
  m[0, 0] = v[0]
  m[1, 1] = v[1]
  m[2, 2] = v[2]
  m[3, 3] = v[3]
  return m


def main():
  ti.init(arch=ti.cuda)

  options = Options()

  window = ti.ui.Window("Display Mesh", (1024, 1024), vsync=True)
  canvas = window.get_canvas()
  scene = window.get_scene()
  camera = ti.ui.Camera()

  camera.up(0, 1, 0)
  camera.position(0, 0, 4)
  # camera.lookat(0, 0, 0)

  hfov = 75
  camera.fov(hfov)

  w, h = window.get_window_shape()

  scene.set_camera(camera)

  sphere_vertices, sphere_indices = sphere_mesh(16)
  splats = random_splats(1)

  transforms = ti.field(dtype=tm.mat4, shape=splats.shape[0])
  splat_transforms(splats, transforms)

  gaussians_3d = Gaussian3D.field(shape=splats.shape[0])


  w, h = window.get_window_shape()
  image = ti.field(dtype=tm.vec3, shape=(w, h))

  while window.running:
      intrinsic = intrinsic_matrix(hfov, window)

      # Flipped to use +z as forward where OpenGL uses -z 
      camera_T_world = diag_matrix([1, 1, -1, 1]) @ tm.mat4(camera.get_view_matrix()).transpose()


      make_checkerboard(image, 32)

      scene.ambient_light([0.2, 0.2, 0.2])
      scene.point_light(pos=[0, 10, 10], color=[1.0, 1.0, 1.0])

      for i in range(splats.shape[0]):
        scene.mesh_instance(sphere_vertices, sphere_indices,  transforms=transforms,
               instance_offset=i, instance_count=1,  color=tuple(splats[i].color))

      camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
      scene.set_camera(camera)

      if options.show_splats:
        draw_splats(splats, image,  intrinsic, camera_T_world, near=0.5)
        canvas.set_image(image)
      elif options.trace_splats:
        to_gaussians_3d(splats, camera_T_world, gaussians_3d)
        trace_splats(gaussians_3d, image,  intrinsic, near=0.5)
      else:
         canvas.scene(scene)


      show_options(window, options)
      window.show()

if __name__=="__main__":
    main()
