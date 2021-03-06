use std::ops::{Add,AddAssign,Sub,Mul,MulAssign,Neg};

use rand::prelude::{ThreadRng,Rng};

use serde::{Deserialize, Serialize};

fn rand_f32(rng: &mut ThreadRng) -> f32 {
    rng.gen::<f32>()
}

#[derive(PartialEq, Debug, Clone, Copy, Deserialize, Serialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }

    pub fn splat(v: f32) -> Self {
        Vec3 { x: v, y: v, z: v }
    }

    #[inline]
    pub fn set(&mut self, x: f32, y: f32, z: f32) {
        self.x = x;
        self.y = y;
        self.z = z;
    }

    #[inline]
    pub fn normalise(&self) -> Self {
        let length = self.len_sq().sqrt();
        self.mul(1.0 / length)
    }

    #[inline]
    pub fn mul_vec(&self, other: Vec3) -> Self {
        Vec3 { x: self.x * other.x, y: self.y * other.y, z: self.z * other.z }
    }

    #[inline]
    pub fn min_vec(&self, other: Vec3) -> Self {
        Vec3 { x: self.x.min(other.x), y: self.y.min(other.y), z: self.z.min(other.z) }
    }

    #[inline]
    pub fn max_vec(&self, other: Vec3) -> Self {
        Vec3 { x: self.x.max(other.x), y: self.y.max(other.y), z: self.z.max(other.z) }
    }

    #[inline]
    pub fn len_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline]
    pub fn map(&self, f: fn(f32) -> f32) -> Self {
        Vec3 {
            x: f(self.x),
            y: f(self.y),
            z: f(self.z),
        }
    }
    
    #[inline]
    pub fn length(&self) -> f32 {
        self.len_sq().sqrt()
    }

    pub fn random_in_unit_sphere(rng: &mut ThreadRng) -> Vec3 {
        let mut p = Vec3::new(2.0, 2.0, 2.0);
        while p.len_sq() >= 1.0 {
            p.set(2.0 * rand_f32(rng) - 1.0, 2.0 * rand_f32(rng) - 1.0, 2.0 * rand_f32(rng) - 1.0);
        }
        p
    }

    pub fn random_in_unit_disk(rng: &mut ThreadRng) -> Vec3 {
        let mut p = Vec3::new(2.0, 2.0, 2.0);
        while p.len_sq() >= 1.0 {
            p.set(2.0 * rand_f32(rng) - 1.0, 2.0 * rand_f32(rng) - 1.0, 0.0);
        }
        p
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Vec3) {
        *self = Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        };
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, scale: f32) -> Vec3 {
        Vec3 {
            x: self.x * scale,
            y: self.y * scale,
            z: self.z * scale,
        }
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, vec: Vec3) -> Vec3 {
        Vec3 {
            x: self * vec.x,
            y: self * vec.y,
            z: self * vec.z,
        }
    }
}

impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, scale: f32) {
        *self = Vec3 {
            x: self.x * scale,
            y: self.y * scale,
            z: self.z * scale,
        };
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

#[inline]
pub fn dot(v1: Vec3, v2: Vec3) -> f32 {
    v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
}

#[inline]
pub fn cross(v1: Vec3, v2: Vec3) -> Vec3 {
    Vec3 {
        x: v1.y * v2.z - v1.z * v2.y,
        y: -(v1.x * v2.z - v1.z * v2.x),
        z: v1.x * v2.y - v1.y * v2.x,
    }
}

#[inline]
pub fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    // v - 2*dot(v,n)*n
    v - 2.0 * dot(v, n) * n
}

#[inline]
pub fn refract(v: Vec3, n: Vec3, ni_over_nt: f32) -> Option<Vec3> {
    let v = v.normalise();
    let dt = dot(v, n);
    let discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
    if discriminant > 0.0 {
        let refracted = ni_over_nt * (v - n*dt) - discriminant.sqrt() * n;
        Some(refracted)
    }
    else {
        None
    }
}

#[derive(Debug, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Ray { origin, direction }
    }
    #[inline]
    pub fn at_t(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_add() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(-1.0, 5.0, 0.0);

        assert_eq!(v1 + v2, Vec3::new(0.0, 7.0, 3.0));
    }

    #[test]
    fn vec3_sub() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(-1.0, 5.0, 0.0);

        assert_eq!(v1 - v2, Vec3::new(2.0, -3.0, 3.0));
    }

    #[test]
    fn vec3_length() {
        // Check len_sq
        assert_eq!(Vec3::new(0.0, 0.0, 0.0).len_sq(), 0.0);
        assert_eq!(Vec3::new(0.0, 1.0, 0.0).len_sq(), 1.0);
        assert_eq!(Vec3::new(0.0, 5.0, 0.0).len_sq(), 25.0);
        assert_eq!(Vec3::new(1.0, 1.0, 1.0).len_sq(), 3.0);

        // Check length
        assert_eq!(Vec3::new(0.0, 0.0, 0.0).length(), 0.0);
        assert_eq!(Vec3::new(0.0, 1.0, 0.0).length(), 1.0);
        assert_eq!(Vec3::new(0.0, 5.0, 0.0).length(), 5.0);
        assert_eq!(Vec3::new(1.0, 1.0, 1.0).length(), f32::sqrt(3.0));
    }

    #[test]
    fn vec3_mul() {
        let v1 = Vec3::new(1.0, 2.0, -3.0);

        assert_eq!(v1 * 3.0, Vec3::new(3.0, 6.0, -9.0));
    }

    #[test]
    fn vec3_normalise() {
        // Normalise an already normalised vector
        let up = Vec3::new(0.0, 1.0, 0.0);
        let normalised = up.normalise();
        assert_eq!(normalised.x, 0.0);
        assert_eq!(normalised.y, 1.0);
        assert_eq!(normalised.z, 0.0);

        // Normalise a longer vector
        let up = Vec3::new(0.0, 3.0, 0.0);
        let normalised = up.normalise();
        assert_eq!(normalised.x, 0.0);
        assert_eq!(normalised.y, 1.0);
        assert_eq!(normalised.z, 0.0);

        // And another
        let up = Vec3::new(0.0, 3.0, 4.0);
        let normalised = up.normalise();
        assert_eq!(normalised.x, 0.0);
        assert_eq!(normalised.y, 3.0 / 5.0);
        assert_eq!(normalised.z, 4.0 / 5.0);
    }

    #[test]
    fn vec3_cross() {
        let x_axis = Vec3::new(1.0, 0.0, 0.0);
        let y_axis = Vec3::new(0.0, 1.0, 0.0);
        let z_axis = Vec3::new(0.0, 0.0, 1.0);

        assert_eq!(cross(x_axis, y_axis), z_axis);
    }

    #[test]
    fn reflect_vector() {
        let x_axis = Vec3::new(1.0, 0.0, 0.0);
        let y_axis = Vec3::new(0.0, 1.0, 0.0);
        let dir = Vec3::new(1.0, -1.0, 0.0); // Within the X-Y plane, heading down at 45 degrees

        assert_eq!(reflect(dir, y_axis), Vec3::new(1.0, 1.0, 0.0));
        assert_eq!(reflect(dir, x_axis), Vec3::new(-1.0, -1.0, 0.0));
    }

    #[test]
    fn refract_vector() {
        let r_glass = 1.5;
        let y_axis = Vec3::new(0.0, 1.0, 0.0);

        // Ray perpindicular to the surface shouldn't bend at all
        let dir = Vec3::new(0.0, -1.0, 0.0);
        assert_eq!(refract(dir, y_axis, 1.0/r_glass), Some(dir));

        // Ray at midway angle (45deg)
        // This angle is not empirically derived, so may be hiding a bug
        let dir = Vec3::new(1.0, -1.0, 0.0);
        let refracted = refract(dir, y_axis, 1.0/r_glass).unwrap();
        assert_eq!(refracted, Vec3::new(0.47140452, -0.8819171, 0.0));
    }

    #[test]
    fn ray_extrapolate() {
        let ray = Ray { origin: Vec3::new(0.0, 0.0, 0.0), direction: Vec3::new(0.0, 0.0, -1.0) };

        assert_eq!(ray.at_t(0.0), Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(ray.at_t(1.0), Vec3::new(0.0, 0.0, -1.0));
        assert_eq!(ray.at_t(2.0), Vec3::new(0.0, 0.0, -2.0));
    }
}
