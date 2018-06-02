
use super::math::*;

pub struct HitRecord<'a> {
    pub t: f32,
    pub p: Vec3,
    pub normal: Vec3,
    pub material: &'a Material,
}

pub enum HitResult<'a> {
    Miss,
    Hit(HitRecord<'a>),
}

pub trait Hitable {
    fn hit<'a>(&'a self, ray: &Ray, t_min: f32, t_max: f32) -> HitResult<'a>;
}

pub struct ScatterResult {
    pub attenuation: Vec3,
    pub scattered: Ray,
}

pub trait Material {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<ScatterResult>;
}

pub struct Invisible {
    // Nothing here
}

impl Material for Invisible {
    fn scatter(&self, _ray: &Ray, _hit: &HitRecord) -> Option<ScatterResult>{
        None
    }
}

pub struct Bounds {
    pub centre: Vec3,
    pub radius: f32,
}

pub struct Sphere {
    pub centre: Vec3,
    pub radius: f32,
    pub material: Box<Material>,
}

fn sphere_ray_intersect(ray: &Ray, t_min: f32, t_max: f32, centre: &Vec3, radius: f32) -> Option<f32> {
    let oc = ray.origin.sub(&centre);
    let a = ray.direction.len_sq();
    let b = 2.0 * dot(&oc, &ray.direction);
    let c = oc.len_sq() - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None
    }

    let d_sqrt = discriminant.sqrt();
    let t1 = (-b - d_sqrt) / (2.0 * a);
    if t1 < t_max && t1 > t_min {
        return Some(t1);
    }

    let t2 = (-b + d_sqrt) / (2.0 * a);
    if t2 < t_max && t2 > t_min {
        return Some(t2);
    }

    None
}

impl Hitable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> HitResult {
        match sphere_ray_intersect(&ray, t_min, t_max, &self.centre, self.radius) {
            Some(t) => {
                let p = ray.at_t(t);
                let normal = p.sub(&self.centre).mul(1.0/self.radius);
                HitResult::Hit(HitRecord { t, p, normal, material: &*self.material })
            },
            None => HitResult::Miss,
        }
    }
}

pub struct Clump {
    pub bounds: Bounds,
    pub objects: Vec<Sphere>,
}

impl Hitable for Clump {
    fn hit<'a>(&'a self, ray: &Ray, t_min: f32, t_max: f32) -> HitResult<'a> {
        // Bounds check for the clump
        // Note the full t range; otherwise the segment is inside completely
        if let None = sphere_ray_intersect(&ray, 0.001, 1000.0, &self.bounds.centre, self.bounds.radius) {
            return HitResult::Miss;
        }

        // Check each contained object
        let mut result = HitResult::Miss;
        let mut closest_so_far = t_max;
        for obj in self.objects.iter() {
            if let HitResult::Hit(record) = obj.hit(&ray, t_min, closest_so_far) {
                closest_so_far = record.t;
                result = HitResult::Hit(record);
            };
        }

        result
    }
}

pub fn hit<'a>(ray: &Ray, t_min: f32, t_max: f32, objects: &'a Vec<Box<Hitable>>) -> HitResult<'a> {
    // // This algorithm seems like the more Rust-like way to do it.
    // // But because it doesn't get to prune future checks based on already seen objects, it is slower.
    // // Perhaps it would be better with multiple threads, or spatially grouped objects
    // match objects.iter()
    //         .map(|obj| obj.hit(&ray, t_min, t_max))
    //         .filter_map(|h| match h { HitResult::Hit(record) => Some(record), _ => None })
    //         .min_by(|h1, h2| h1.t.partial_cmp(&h2.t).unwrap()) {
    //     Some(record) => HitResult::Hit(record),
    //     None => HitResult::Miss,
    // }

    let mut result = HitResult::Miss;
    let mut closest_so_far = t_max;
    for obj in objects {
        match (*obj).hit(&ray, t_min, closest_so_far) {
            HitResult::Hit(record) => {
                closest_so_far = record.t;
                result = HitResult::Hit(record);
            },
            HitResult::Miss => {
            }
        };
    }

    result
}

#[cfg(test)]
mod tests {
    use super::super::math::*;
    use super::*;
    use super::HitResult::*;

    #[test]
    fn hit_sphere_works() {
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let left = Vec3::new(-1.0, 0.0, 0.0);
        let down_y = Ray { origin: origin.clone(), direction: Vec3::new(0.0, -1.0, 0.0) };
        let down_y_parallel = Ray { origin: left.mul(2.0), direction: Vec3::new(0.0, -1.0, 0.0) };
        // Expected hit: ray along y axis and sphere 2 units down y axis
        let sphere = Sphere { centre: Vec3::new(0.0, -2.0, 0.0), radius: 1.0, material: Box::new(Invisible {}) };
        match sphere.hit(&down_y, 0.0, 1000.0) {
            Miss => panic!("This ray and sphere were supposed to hit"),
            Hit(record) => assert_eq!(record.t, 1.0),
        };
        // Expected miss: ray parallel to y axis and sphere 2 units down y axis
        match sphere.hit(&down_y_parallel, 0.0, 1000.0) {
            Miss => (),
            Hit(_) => panic!("This ray and sphere were supposed to miss"),
        };
    }
}
