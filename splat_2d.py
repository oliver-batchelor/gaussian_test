from typing import Tuple
import taichi as ti
import taichi.math as tm

from camera import Camera
from splat_3d import AABox, Splat3D, to_covariance
from transform import quat_to_mat

@ti.func
def cov_to_conic_radius(
    cov: tm.mat2,
): # -> Tuple[tm.vec3, ti.f32]:

  det_cov = cov.determinant()
  inv_cov = (1. / det_cov) * \
      tm.mat2([[cov[1, 1], -cov[0, 1]],
                    [-cov[1, 0], cov[0, 0]]])
  
  conic = tm.vec3([inv_cov[0, 0], inv_cov[0, 1], inv_cov[1, 1]])

  mid = 0.5 * (cov[0, 0] + cov[1, 1])
  lambda1 = mid + ti.sqrt(max(0.1, mid * mid - det_cov))
  lambda2 = mid - ti.sqrt(max(0.1, mid * mid - det_cov))
  radius =  3.0 * ti.sqrt(max(lambda1, lambda2))

  return (conic, radius)



@ti.func
def density_from_conic(
    xy: ti.math.vec2,
    gaussian_mean: ti.math.vec2,
    conic : ti.math.vec3,
) -> ti.f32:
    d = xy - gaussian_mean
    exponent = -0.5 * (d.x * d.x * conic.x + d.y * d.y * conic.z) \
        - d.x * d.y * conic.y
    return ti.exp(exponent)




@ti.dataclass
class Splat2D:
    uv: ti.math.vec2
    depth : ti.f32
    
    conic : ti.math.vec3
    radius : ti.f32

    color: ti.math.vec3
    opacity: ti.f32

        
    @ti.func
    def from_vec(self, v:ti.template()):
        self.uv = v[0:2]
        self.conic = v[2:5]
        self.radius = v[5]
        self.color = v[6:9]
        self.opacity = v[9]


@ti.func
def project_splat(
    splat: Splat3D,
    camera: Camera,
    world_to_camera: tm.mat4,
) -> Splat2D:
      
  p = (world_to_camera @ tm.vec4(splat.p, 1.0)).xyz
  f = camera.focal_length()

  J = tm.mat3(
    f.x / p.z, 0.0, -(f.y * p.x) / (p.z * p.z),
    0.0, f.y / p.z, -(f.y * p.y) / (p.z * p.z),
    0, 0, 0)
  
  scaling = ti.Matrix.cols([splat.scale, splat.scale, splat.scale])

  axes = scaling * (world_to_camera[:3, :3] @ quat_to_mat(splat.q))


  # J @ axes @ axes^T @ J^T
  m = J @ axes 
  cov_2d = m @ m.transpose() 

  uv = camera.image_t_camera @ p

  conic, radius = cov_to_conic_radius(cov_2d[:2,:2])

  return Splat2D(uv.xy / uv.z, uv.z, conic, radius, splat.color, splat.opacity)
