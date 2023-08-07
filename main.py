from dataclasses import dataclass
from typing import Tuple
import taichi as ti
import taichi.math as tm
import numpy as np

import open3d as o3d
from camera import Camera

from rand import random_splats
from scipy.spatial.transform import Rotation as R
from splat_2d import Splat2D, density_from_conic, project_splat

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


def show_options(window, options):

    window.GUI.begin("Display Panel", 0.05, 0.1, 0.2, 0.15)
    # display_mode = window.GUI.slider_int("Value Range", some_int_type_value, 0, 5)
    options.show_splats = window.GUI.checkbox("Show Splats", options.show_splats)
    window.GUI.end()



@ti.kernel
def splat_transforms(splats:ti.template(), transforms:ti.template()):
  for i in range(splats.shape[0]):
    splat = splats[i]
    rotation = quat_to_mat(splat.q)

    m = tm.mat4(0)
    m[0:3, 0:3] = rotation @ scaling(splat.scale)
    m[0:3, 3] = splat.p
    m[3, 3] = 1.0

    transforms[i] = m

    
def intrinsic_matrix(hfov, window):
    width, height = window.get_window_shape()

    fx = 1.0 / np.tan(hfov * 0.5 * np.pi / 180.0)
    fy = fx * (height / width)
    cx = width * 0.5
    cy = height * 0.5

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ])
   
@ti.kernel
def draw_splats(splats:ti.template(), image:ti.template(), intrinsic:tm.mat3, view:tm.mat4):
  camera = Camera(image.shape[:2], intrinsic)

  for p in ti.grouped(ti.ndrange(image.shape[0], image.shape[1])):
    image[p] = 0.0

  for i in splats:
    splat2d = project_splat(splats[i], camera, view)
    if splat2d.depth < 0.0:
      continue

    radius = 50
    lower = ti.max(ti.floor(splat2d.uv - radius, ti.i32), 0)
    upper = ti.min(ti.ceil(splat2d.uv + radius, ti.i32), tm.ivec2(image.shape[:2]) - 1)
    # print(i, splat2d.uv, splat2d.radius, upper-lower)
    
    for x, y in ti.ndrange((lower.x, lower.y), (upper.x, upper.y)):
      xy = ti.math.vec2(x, y)
      density = density_from_conic(xy, splat2d.uv, splat2d.conic)

      image[x, y] = splat2d.color #+= splat2d.color * density * splat2d.opacity

def main():
  ti.init(arch=ti.cuda, debug=True)

  options = Options()

  window = ti.ui.Window("Display Mesh", (1024, 1024), vsync=True)
  canvas = window.get_canvas()
  scene = window.get_scene()
  camera = ti.ui.Camera()

  camera.up(0, 1, 0)
  camera.position(0, 0, 10)
  # camera.lookat(0, 0, 0)

  hfov = 60
  camera.fov(hfov)

  w, h = window.get_window_shape()

  scene.set_camera(camera)

  sphere_vertices, sphere_indices = sphere_mesh(16)
  splats = random_splats(10)

  transforms = ti.field(dtype=tm.mat4, shape=splats.shape[0])
  splat_transforms(splats, transforms)


  w, h = window.get_window_shape()
  image = ti.field(dtype=tm.vec3, shape=(w, h))

  while window.running:
      intrinsic = intrinsic_matrix(hfov, window)

      draw_splats(splats, image,  intrinsic, camera.get_view_matrix())

      scene.ambient_light([0.2, 0.2, 0.2])
      scene.point_light(pos=[0, 10, 10], color=[1.0, 1.0, 1.0])

      for i in range(splats.shape[0]):
        scene.mesh_instance(sphere_vertices, sphere_indices,  transforms=transforms,
               instance_offset=i, instance_count=1,  color=tuple(splats[i].color))

      camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
      scene.set_camera(camera)

      if options.show_splats:
        canvas.set_image(image)
      else:
         canvas.scene(scene)


      show_options(window, options)
      window.show()

if __name__=="__main__":
    main()
