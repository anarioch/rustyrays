
use super::math::{Vec3,Ray};

pub enum HitResult {
    Miss,
    Hit { t: f32, p: Vec3, normal: Vec3 },
}

pub trait Hitable {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> HitResult;
}

pub struct Sphere {
    pub centre: Vec3,
    pub radius: f32,
}

impl Hitable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> HitResult {
        let oc = ray.origin.sub(&self.centre);
        let a = ray.direction.len_sq();
        let b = 2.0 * oc.dot(&ray.direction);
        let c = oc.len_sq() - self.radius * self.radius;
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            HitResult::Miss
        }
        else {
            let d_sqrt = discriminant.sqrt();
            let t1 = (-b - d_sqrt) / (2.0 * a);
            let t2 = (-b + d_sqrt) / (2.0 * a);
            if t1 < t_max && t1 > t_min {
                let p = ray.at_t(t1);
                let normal = p.sub(&self.centre).mul(1.0/self.radius);
                HitResult::Hit { t: t1, p, normal }
            }
            else if t2 < t_max && t2 > t_min {
                let p = ray.at_t(t2);
                let normal = p.sub(&self.centre).mul(1.0/self.radius);
                HitResult::Hit { t: t2, p, normal }
            }
            else {
                HitResult::Miss
            }
        }
    }
}

pub fn hit(ray: &Ray, t_min: f32, t_max: f32, objects: &Vec<Box<Hitable>>) -> HitResult {
    let mut result = HitResult::Miss;
    let mut closest_so_far = t_max;
    for obj in objects {
        match (*obj).hit(&ray, 0.0, closest_so_far) {
            HitResult::Hit { t, p, normal } => {
                closest_so_far = t;
                result = HitResult::Hit { t, p, normal };
            },
            HitResult::Miss => {
            }
        };
    }

    result
}

#[cfg(test)]
mod tests {
    use super::super::math::{Vec3,Ray};
    use super::{Sphere,Hitable};
    use super::HitResult::*;

    #[test]
    fn hit_sphere_works() {
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let left = Vec3::new(-1.0, 0.0, 0.0);
        let down_y = Ray { origin: origin.clone(), direction: Vec3::new(0.0, -1.0, 0.0) };
        let down_y_parallel = Ray { origin: left.mul(2.0), direction: Vec3::new(0.0, -1.0, 0.0) };
        // Expected hit: ray along y axis and sphere 2 units down y axis
        let sphere = Sphere { centre: Vec3::new(0.0, -2.0, 0.0), radius: 1.0 };
        match sphere.hit(&down_y, 0.0, 1000.0) {
            Miss => panic!("This ray and sphere were supposed to hit"),
            Hit{ t, p: _, normal: _ } => assert_eq!(t, 1.0),
        };
        // Expected miss: ray parallel to y axis and sphere 2 units down y axis
        match sphere.hit(&down_y_parallel, 0.0, 1000.0) {
            Miss => (),
            Hit { t: _, p: _, normal: _ } => panic!("This ray and sphere were supposed to miss"),
        };
        // assert_eq!(hit_sphere(&origin, 1.0, &down_z_parallel), SphereHitResult::Miss);
    }
}
