from typing import Tuple
import taichi as ti
import taichi.math as tm
import numpy as np

import open3d as o3d
from camera import Camera

from rand import random_splats
from scipy.spatial.transform import Rotation as R
from splat_2d import project_splat

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


def show_options(window):
    some_int_type_value = 0

    window.GUI.begin("Display Panel", 0.05, 0.1, 0.2, 0.15)
    display_mode = window.GUI.slider_int("Value Range", some_int_type_value, 0, 5)
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

    
def intrinsic_matrix(camera, window):
    width, height = window.get_window_shape()
    fx = 1.0 / np.tan(camera.fov() * 0.5 * np.pi / 180.0)
    fy = fx * width / height
    cx = width * 0.5
    cy = height * 0.5

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ])
   
@ti.kernel
def draw_splats(splats:ti.template(), image:ti.template(), intrinsic:tm.mat3, view:tm.mat4):
  camera = Camera(image.shape, intrinsic)

  for splat3d in splats:
    splat2d = project_splat(splat3d, camera, view)
    lower = ti.floor(splat2d.p - splat2d.radius, ti.i32)
    upper = ti.ceil(splat2d.p + splat2d.radius, ti.i32)
   
    for x, y in ti.ndrange(lower, upper):
      xy = ti.math.vec2(x, y)
      density = density_from_conic(xy, splat2d.p, splat2d.conic)
      image[x, y] += splat2d.color * density * splat2d.opacity

def main():
  ti.init(arch=ti.cuda)

  window = ti.ui.Window("Display Mesh", (1024, 1024), vsync=True)
  canvas = window.get_canvas()
  scene = window.get_scene()
  camera = ti.ui.Camera()

  camera.up(0, 1, 0)
  camera.position(0, 0, 10)
  # camera.lookat(0, 0, 0)
  camera.fov(60)

  scene.set_camera(camera)

  sphere_vertices, sphere_indices = sphere_mesh(16)
  splats = random_splats(10)

  transforms = ti.field(dtype=tm.mat4, shape=splats.shape[0])
  splat_transforms(splats, transforms)

  w, h = window.get_window_shape()
  image = ti.field(dtype=ti.f32, shape=(w, h, 3))

  while window.running:
      intrinsic = intrinsic_matrix(camera, window)

      draw_splats(splats, image,  intrinsic, camera.get_view_matrix())

      scene.ambient_light([0.2, 0.2, 0.2])
      scene.point_light(pos=[0, 10, 10], color=[1.0, 1.0, 1.0])

      for i in range(splats.shape[0]):
        scene.mesh_instance(sphere_vertices, sphere_indices,  transforms=transforms,
               instance_offset=i, instance_count=1,  color=tuple(splats[i].color))

      camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
      scene.set_camera(camera)




      canvas.scene(scene)
      show_options(window)
      window.show()

if __name__=="__main__":
    main()
