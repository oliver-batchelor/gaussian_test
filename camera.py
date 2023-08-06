import taichi as ti
import taichi.math as tm


@ti.dataclass
class Camera:
    image_size : tm.ivec2
    image_t_camera : tm.mat3

    
    @ti.func
    def focal_length(self):
        return tm.vec2(self.image_t_camera[0, 0], self.image_t_camera[1, 1])