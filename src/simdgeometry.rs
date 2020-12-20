#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

macro_rules! _mm_shuffle {
    ($z:expr, $y:expr, $x:expr, $w:expr) => {
        ($z << 6) | ($y << 4) | ($x << 2) | $w
    };
}

#[inline]
pub fn new_pos(x: f32, y: f32, z: f32) -> __m128 {
    unsafe {
        _mm_set_ps(1.0, z, y, x)
    }
}

#[inline]
pub fn new_dir(x: f32, y: f32, z: f32) -> __m128 {
    unsafe {
        _mm_set_ps(0.0, z, y, x)
    }
}

#[inline]
pub fn vec_x(v: __m128) -> f32 {
    unsafe {
        _mm_cvtss_f32(v)
    }
}

#[inline]
pub fn vec_y(v: __m128) -> f32 {
    unsafe {
        _mm_cvtss_f32(_mm_shuffle_ps(v, v, _mm_shuffle!(0, 0, 0, 1)))
    }
}

#[inline]
pub fn vec_z(v: __m128) -> f32 {
    unsafe {
        _mm_cvtss_f32(_mm_shuffle_ps(v, v, _mm_shuffle!(0, 0, 0, 2)))
    }
}

/// Dot product for 3-component vector (which is all this library uses)
#[inline]
pub fn dot(a: __m128, b: __m128) -> f32 {
    unsafe {
        _mm_cvtss_f32(_mm_dp_ps(a, b, 0x77))
    }
}

/// Cross product for a 3D vector (ignores fourth component)
#[inline]
pub fn cross(a: __m128, b: __m128) -> __m128 {
    unsafe {
        let a_yzx = _mm_shuffle_ps(a, a, _mm_shuffle!(3, 0, 2, 1));
        let b_yzx = _mm_shuffle_ps(b, b, _mm_shuffle!(3, 0, 2, 1));
        let c = _mm_sub_ps(_mm_mul_ps(a, b_yzx), _mm_mul_ps(a_yzx, b));
        _mm_shuffle_ps(c, c, _mm_shuffle!(3, 0, 2, 1))
    }
}

#[inline]
pub fn vec_len_sq(v: __m128) -> f32 {
    unsafe {
        _mm_cvtss_f32(_mm_dp_ps(v, v, 0x77))
    }
}

#[inline]
pub fn vec_len(v: __m128) -> f32 {
    unsafe {
        _mm_cvtss_f32(_mm_sqrt_ps(_mm_dp_ps(v, v, 0x77)))
    }
}

#[inline]
pub fn unit_vec(v: __m128) -> __m128 {
    unsafe {
        let len = _mm_sqrt_ps(_mm_dp_ps(v, v, 0x77));
        _mm_div_ps(v, len)
    }
}

#[inline]
pub fn reflect(v: __m128, n: __m128) -> __m128 {
    // v - 2.0 * dot(v, n) * n
    unsafe {
        let two = _mm_set1_ps(2.0);
        let dot_vn = _mm_dp_ps(v, n, 0x77);
        let c = _mm_mul_ps(_mm_mul_ps(two, dot_vn), n);
        _mm_sub_ps(v, c)
    }
}

#[inline]
pub fn refract(v: __m128, n: __m128, ni_over_nt: f32) -> Option<__m128> {
    unsafe {
        let v = unit_vec(v);
        let dot_vn = _mm_dp_ps(v, n, 0x77);
        let dt = _mm_cvtss_f32(dot_vn);
        let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
        if discriminant > 0.0 {
            // let refracted = ni_over_nt * (v - n*dt) - discriminant.sqrt() * n;
            let a = _mm_fmsub_ps(dot_vn, n, v); // Note this subtracts backwards, which is fixed by negating ni_over_nt later
            let sqrt_d = _mm_sqrt_ps(_mm_set1_ps(discriminant));
            let b = _mm_mul_ps(sqrt_d, n);
            let refracted = _mm_fmsub_ps(_mm_set1_ps(-ni_over_nt), a, b);
            Some(refracted)
        }
        else {
            None
        }
    }
}

pub fn vec_eq(a: __m128, b: __m128) -> bool {
    unsafe {
        let cmp = _mm_cmpeq_ps(a, b);
        let mask = _mm_movemask_ps(cmp);
        mask & 0x7 == 0x7 // Compare only the key 3 components
    }
}

pub fn vec_approx_eq(a: __m128, b: __m128, epsilon: f32) -> bool {
    unsafe {
        let eps = _mm_set1_ps(epsilon);
        // Compute absolute difference
        let diff = _mm_andnot_ps(_mm_set1_ps(-0.0), _mm_sub_ps(a, b));
        // Check whether all components are less than epsilon
        let cmp = _mm_cmplt_ps(diff, eps);
        let mask = _mm_movemask_ps(cmp);
        mask & 0x7 == 0x7 // Compare only the key 3 components
    }
}

#[derive(Debug)]
pub struct Ray {
    pub origin: __m128,
    pub direction: __m128,
}

impl Ray {
    pub fn new(origin: __m128, direction: __m128) -> Self {
        Ray { origin, direction }
    }
    #[inline]
    pub fn at_t(&self, t: f32) -> __m128 {
        // origin + t * direction
        unsafe {
            _mm_fmadd_ps(_mm_set1_ps(t), self.direction, self.origin)
        }
    }
}

/// Axis Aligned Bounding Box
#[derive(Debug)]
pub struct AABB {
    pub min: __m128,
    pub max: __m128,
}

/// Horizontal min of a 3-component vector
unsafe fn _hmin_ps(v: __m128) -> f32 {
    // Shuffle y and z values into x component of tmp vectors
    let y = _mm_shuffle_ps(v, v, _mm_shuffle!(0,0,0,1));
    let z = _mm_shuffle_ps(v, v, _mm_shuffle!(0,0,0,2));
    // Then do two mins to compare x / y / z
    let min1 = _mm_min_ps(v, y);
    let min2 = _mm_min_ps(min1, z);
    _mm_cvtss_f32(min2)
}

/// Horizontal min of a 3-component vector
unsafe fn _hmax_ps(v: __m128) -> f32 {
    // Shuffle y and z values into x component of tmp vectors
    let y = _mm_shuffle_ps(v, v, _mm_shuffle!(0,0,0,1));
    let z = _mm_shuffle_ps(v, v, _mm_shuffle!(0,0,0,2));
    // Then do two mins to compare x / y / z
    let max1 = _mm_max_ps(v, y);
    let max2 = _mm_max_ps(max1, z);
    _mm_cvtss_f32(max2)
}

impl AABB {
    pub fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> bool {
        unsafe {
            // Turn min/max into vectors
            let t_min = _mm_set1_ps(t_min);
            let t_max = _mm_set1_ps(t_max);

            // Swap around min/max boundaries on an axis if ray is moving negatively in it
            let mask = _mm_cmplt_ps(ray.direction, _mm_setzero_ps());
            let min = _mm_or_ps(_mm_and_ps(mask, self.max),
                                _mm_andnot_ps(mask, self.min));
            let max = _mm_or_ps(_mm_and_ps(mask, self.min),
                                _mm_andnot_ps(mask, self.max));

            // Compute where the entry/exit points on each axis
            let inv_d = _mm_div_ps(_mm_set1_ps(1.0), ray.direction);
            let box_min = _mm_mul_ps(_mm_sub_ps(min, ray.origin), inv_d);
            let box_max = _mm_mul_ps(_mm_sub_ps(max, ray.origin), inv_d);

            // Clamp them to the input min/max values
            let t_min = _mm_max_ps(box_min, t_min);
            let t_max = _mm_min_ps(box_max, t_max);

            // let t_min = hmax_ps(t_min);
            // let t_max = hmin_ps(t_max);

            // Find the highest element of the min vector and lowest of the max vector - the horizontal min/max
            // This checks only x/y/z components and interleaves operations to save the number of shuffles
            let tmp1 = _mm_shuffle_ps(t_min, t_max, _mm_shuffle!(1,0,3,2)); // min.z min.w max.x max.y
            let tmp2 = _mm_shuffle_ps(t_min, t_max, _mm_shuffle!(1,1,1,1)); // min.y _ max.y _
            let min1 = _mm_max_ps(t_min, tmp1);                         // xz yw o o 
            let max1 = _mm_min_ps(t_max, tmp1);                         // o o zx wy
            let min2 = _mm_max_ps(min1, tmp2);                          // xzy _ _ _
            let max2 = _mm_min_ps(max1, tmp2);                          // _ _ zxy _
            let max3 = _mm_shuffle_ps(max2, max2, _mm_shuffle!(2,2,2,2)); // zxy _ _ _
            let t_min = _mm_cvtss_f32(min2);
            let t_max = _mm_cvtss_f32(max3);

            t_max > t_min
        }
    }

    pub fn union(&self, other: &Self) -> Self {
        AABB {
            min: unsafe { _mm_min_ps(self.min, other.min) },
            max: unsafe { _mm_max_ps(self.max, other.max) },
        }
    }

    pub fn union_assign(&mut self, other: &Self) {
        self.min = unsafe { _mm_min_ps(self.min, other.min) };
        self.max = unsafe { _mm_max_ps(self.max, other.max) };
    }
}

fn _vec_sub(a: __m128, b: __m128) -> __m128 {
    unsafe {
        _mm_sub_ps(a, b)
    }
}

fn _sphere_ray_intersect(ray: &Ray, _t_min: f32, _t_max: f32, centre: __m128, radius: f32) -> Option<f32> {
    let ray_pos = ray.origin;
    let ray_dir = ray.direction;
    // TODO: This code won't be helped by vectorisation as is.
    //       I think it may need to operate on 4 spheres at once, in order to parallelise the scalar ops
    unsafe {
        // This part is using proper vector operations, but storing the scalar result in all channels of the result 
        let oc = _mm_sub_ps(ray_pos, centre);
        let a = _mm_dp_ps(ray_dir, ray_dir, 0x77);
        let b = _mm_mul_ps(_mm_set1_ps(2.0), _mm_dp_ps(oc, ray_dir, 0x77));
        let rad_sq = _mm_set1_ps(radius * radius);
        let c = _mm_sub_ps(_mm_dp_ps(oc, oc, 0x77), rad_sq);
        // This part is scalar math, but doing the same calculation in each vector channel
        // TODO: Implement four-sphere version of this function
        // let discriminant = b * b - 4.0 * a * c;
        let four_ac = _mm_mul_ps(a, c);
        let four_ac = _mm_mul_ps(_mm_set1_ps(4.0), four_ac);
        let _discriminant = _mm_fmsub_ps(b, b, four_ac);
        // TODO: Translate these scalar ops to SIMD and re-enable
        // if discriminant < 0.0 {
        //     return None
        // }

        // let d_sqrt = discriminant.sqrt();
        // let t1 = (-b - d_sqrt) / (2.0 * a);
        // if t1 < t_max && t1 > t_min {
        //     return Some(t1);
        // }

        // let t2 = (-b + d_sqrt) / (2.0 * a);
        // if t2 < t_max && t2 > t_min {
        //     return Some(t2);
        // }

        None
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_length() {
        // Check len_sq
        assert_eq!(vec_len_sq(new_dir(0.0, 0.0, 0.0)), 0.0);
        assert_eq!(vec_len_sq(new_dir(0.0, 1.0, 0.0)), 1.0);
        assert_eq!(vec_len_sq(new_dir(0.0, 5.0, 0.0)), 25.0);
        assert_eq!(vec_len_sq(new_dir(1.0, 1.0, 1.0)), 3.0);

        // Check length
        assert_eq!(vec_len(new_dir(0.0, 0.0, 0.0)), 0.0);
        assert_eq!(vec_len(new_dir(0.0, 1.0, 0.0)), 1.0);
        assert_eq!(vec_len(new_dir(0.0, 5.0, 0.0)), 5.0);
        assert_eq!(vec_len(new_dir(1.0, 1.0, 1.0)), f32::sqrt(3.0));
    }

    #[test]
    fn vec_equality() {
        assert!(vec_eq(new_pos(0.0, 1.0, 2.0), new_pos(0.0, 1.0, 2.0)));
        assert!(!vec_eq(new_pos(0.0, 1.0, 2.0), new_pos(2.0, 1.0, 2.0)));

        assert!(vec_approx_eq(new_pos(0.0, 1.0, 2.01), new_pos(0.0, 1.0, 2.0), 1e-1));
        assert!(!vec_approx_eq(new_pos(0.0, 1.0, 2.01), new_pos(0.0, 1.0, 2.0), 1e-3));
    }

    #[test]
    fn vec3_normalise() {
        let up = new_dir(0.0, 1.0, 0.0);
        unsafe {
        let inv_len = _mm_div_ps(_mm_set1_ps(1.0), _mm_sqrt_ps(_mm_dp_ps(up, up, 0x77)));
        assert_eq!(_mm_cvtss_f32(inv_len), 1.0);
        }

        // Normalise an already normalised vector
        let up = new_dir(0.0, 1.0, 0.0);
        let normalised = unit_vec(up);
        assert!(vec_eq(normalised, new_dir(0.0, 1.0, 0.0)));

        // Normalise a longer vector
        let up = new_dir(0.0, 3.0, 0.0);
        let normalised = unit_vec(up);
        println!("normalised = {:?}", normalised);
        assert!(vec_eq(normalised, new_dir(0.0, 1.0, 0.0)));

        // And another
        let up = new_dir(0.0, 3.0, 4.0);
        let normalised = unit_vec(up);
        assert!(vec_eq(normalised, new_dir(0.0, 3.0 / 5.0, 4.0 / 5.0)));
    }

    #[test]
    fn vec3_cross() {
        let x_axis = new_dir(1.0, 0.0, 0.0);
        let y_axis = new_dir(0.0, 1.0, 0.0);
        let z_axis = new_dir(0.0, 0.0, 1.0);

        assert!(vec_eq(cross(x_axis, y_axis), z_axis));
    }

    #[test]
    fn reflect_vector() {
        let x_axis = new_dir(1.0, 0.0, 0.0);
        let y_axis = new_dir(0.0, 1.0, 0.0);
        let dir = new_dir(1.0, -1.0, 0.0); // Within the X-Z plane, heading down at 45 degrees

        assert!(vec_approx_eq(reflect(dir, y_axis), new_dir(1.0, 1.0, 0.0), 1e-15));
        assert!(vec_approx_eq(reflect(dir, x_axis), new_dir(-1.0, -1.0, 0.0), 1e-15));
    }

    #[test]
    fn refract_vector() {
        let r_glass = 1.5;
        let y_axis = new_dir(0.0, 1.0, 0.0);

        // Ray perpindicular to the surface shouldn't bend at all
        let dir = new_dir(0.0, -1.0, 0.0);
        assert!(vec_approx_eq(refract(dir, y_axis, 1.0/r_glass).unwrap(), dir, 1e-15));

        // Ray at midway angle (45deg)
        // This angle is not empirically derived, so may be hiding a bug
        let dir = new_dir(1.0, -1.0, 0.0);
        let refracted = refract(dir, y_axis, 1.0/r_glass).unwrap();
        println!("refracted = {:?}", refracted);
        assert!(vec_approx_eq(refracted, new_dir(0.47140452, -0.8819171, 0.0), 1e-15));
    }

    #[test]
    fn ray_extrapolate() {
        let ray = Ray { origin: new_pos(0.0, 0.0, 0.0), direction: new_dir(0.0, 0.0, -1.0) };

        assert!(vec_approx_eq(ray.at_t(0.0), new_dir(0.0, 0.0, 0.0), 1e-15));
        assert!(vec_approx_eq(ray.at_t(1.0), new_dir(0.0, 0.0, -1.0), 1e-15));
        assert!(vec_approx_eq(ray.at_t(2.0), new_dir(0.0, 0.0, -2.0), 1e-15));
    }

    #[test]
    fn ray_aabb_hit() {
        // Given: An AABB
        let aabb = AABB { min: new_pos( -2.0, -2.0, -2.0), max: new_pos(2.0, -1.5, 2.0) };
        let origin = new_pos(0.0, 0.0, 0.0);
        let along_x = new_pos(3.0, 0.0, 0.0);
        let y_axis = new_dir(0.0, 1.0, 0.0);
        let ny_axis = new_dir(0.0, -1.0, 0.0);

        // Temp test: construct the ray separately to allow debugging
        let ray = Ray { origin: origin, direction: ny_axis };
        assert_eq!(aabb.hit(&ray, 0.0, 1000.0), true);

        // When: we check for a hit then each ray raturns appropriately
        assert_eq!(aabb.hit(&Ray { origin: origin,  direction: ny_axis }, 0.0, 1000.0), true);  // ray pointing into box
        assert_eq!(aabb.hit(&Ray { origin: origin,  direction: ny_axis }, 0.0,    1.0), false); // ray pointing into box but t range too short
        assert_eq!(aabb.hit(&Ray { origin: origin,  direction:  y_axis }, 0.0, 1000.0), false); // ray pointing away from box
        assert_eq!(aabb.hit(&Ray { origin: along_x, direction: ny_axis }, 0.0, 1000.0), false); // ray parallel to y axis and along x
        assert_eq!(aabb.hit(&Ray { origin: along_x, direction:  y_axis }, 0.0, 1000.0), false); // ray parallel to y axis and along x, pointing away
    }
}
