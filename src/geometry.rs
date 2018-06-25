
use super::math::*;
use super::materials::Material;

use std::cmp::Ordering;
use rand;
use rand::Rng;

pub fn new_pos(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

pub fn new_dir(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

/// Axis Aligned Bounding Box
#[derive(Debug)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn hit(&self, ray: &Ray, mut t_min: f32, mut t_max: f32) -> bool {
        let inv_d = ray.direction.map(|x| 1.0 / x);
        let t0 = (self.min - ray.origin).mul_vec(inv_d);
        let t1 = (self.max - ray.origin).mul_vec(inv_d);

        if inv_d.x < 0.0 {
            t_min = t_min.max(t1.x);
            t_max = t_max.min(t0.x);
            if t_max <= t_min {
                return false;
            }
        }
        else {
            t_min = t_min.max(t0.x);
            t_max = t_max.min(t1.x);
            if t_max <= t_min {
                return false;
            }
        }

        if inv_d.y < 0.0 {
            t_min = t_min.max(t1.y);
            t_max = t_max.min(t0.y);
            if t_max <= t_min {
                return false;
            }
        }
        else {
            t_min = t_min.max(t0.y);
            t_max = t_max.min(t1.y);
            if t_max <= t_min {
                return false;
            }
        }

        if inv_d.z < 0.0 {
            t_min = t_min.max(t1.z);
            t_max = t_max.min(t0.z);
            if t_max <= t_min {
                return false;
            }
        }
        else {
            t_min = t_min.max(t0.z);
            t_max = t_max.min(t1.z);
            if t_max <= t_min {
                return false;
            }
        }

        // let (xt0,xt1) = if inv_d.x < 0.0 { (t1.x,t0.x) } else { (t0.x,t1.x) };
        // let t_min = t_min.max(xt0);
        // let t_max = t_max.min(xt1);
        // if t_max <= t_min {
        //     return false;
        // }

        // let (yt0,yt1) = if inv_d.y < 0.0 { (t1.y,t0.y) } else { (t0.y,t1.y) };
        // let t_min = t_min.max(yt0);
        // let t_max = t_max.min(yt1);
        // if t_max <= t_min {
        //     return false;
        // }

        // let (zt0,zt1) = if inv_d.z < 0.0 { (t1.z,t0.z) } else { (t0.z,t1.z) };
        // let t_min = t_min.max(zt0);
        // let t_max = t_max.min(zt1);
        // if t_max <= t_min {
        //     return false;
        // }

        true
    }

    pub fn union(&self, other: &AABB) -> AABB {
        AABB {
            min: self.min.min_vec(other.min),
            max: self.max.max_vec(other.max),
        }
    }

    pub fn union_assign(&mut self, other: &AABB) {
        self.min = self.min.min_vec(other.min);
        self.max = self.max.max_vec(other.max);
    }
}

/// A record of where a ray hit an object, including a reference to the material
pub struct HitRecord<'a> {
    pub t: f32,
    pub p: Vec3,
    pub normal: Vec3,
    pub material: &'a dyn Material,
}

pub trait Hitable {
    fn hit<'a>(&'a self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'a>>;
    fn bounds(&self) -> Option<AABB>;
}

pub struct Sphere {
    pub centre: Vec3,
    pub radius: f32,
    pub material: Box<dyn Material>,
}

fn sphere_ray_intersect(ray: &Ray, t_min: f32, t_max: f32, centre: Vec3, radius: f32) -> Option<f32> {
    let oc = ray.origin - centre;
    let a = ray.direction.len_sq();
    let b = 2.0 * dot(oc, ray.direction);
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
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        match sphere_ray_intersect(&ray, t_min, t_max, self.centre, self.radius) {
            Some(t) => {
                let p = ray.at_t(t);
                let normal = (p - self.centre) * (1.0/self.radius);
                Some(HitRecord { t, p, normal, material: &*self.material })
            },
            None => None,
        }
    }
    fn bounds(&self) -> Option<AABB> {
        let rad = new_dir(self.radius, self.radius, self.radius);
        Some(AABB { min: self.centre - rad, max: self.centre + rad })
    }
}

pub enum AARectWhich {
    XY,
    XZ,
    YZ,
}
pub struct AARect {
    pub which: AARectWhich,
    pub a_min: f32,
    pub a_max: f32,
    pub b_min: f32,
    pub b_max: f32,
    pub c: f32,
    pub negate_normal: bool,
    pub material: Box<dyn Material>,
}

impl Hitable for AARect {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        // Swizzle the inputs to match an XY plane layout
        let origin = ray.origin;
        let direction = ray.direction;
        let (origin, direction) = match self.which {
            AARectWhich::XY => (origin, direction),
            AARectWhich::XZ => (new_pos(origin.x, origin.z, origin.y), new_dir(direction.x, direction.z, direction.y)),
            AARectWhich::YZ => (new_pos(origin.y, origin.z, origin.x), new_dir(direction.y, direction.z, direction.x)),
        };

        // Calculate ray/plane intersect and bail if it is outside the required t range
        let t = (self.c - origin.z) / direction.z;
        if t < t_min || t > t_max {
            return None;
        }

        // Determine where in the plane the intersection is and bail if it is outside the rectangle
        let x = origin.x + t * direction.x;
        let y = origin.y + t * direction.y;
        if x < self.a_min || x > self.a_max ||
           y < self.b_min || y > self.b_max {
            return None;
        }

        let p = ray.at_t(t);
        let normal = new_dir(0.0, 0.0, if self.negate_normal { -1.0 } else { 1.0 });
        Some(HitRecord { t, p, normal, material: &*self.material })
    }
    fn bounds(&self) -> Option<AABB> {
        const FUDGE: f32 = 0.005; // Give the infinitesimal plane a pretend width for bounds calculations
        match self.which {
            AARectWhich::XY => Some(AABB { min: new_pos(self.a_min, self.b_min, self.c - FUDGE), max: new_pos(self.a_max, self.b_max, self.c + FUDGE) }),
            AARectWhich::XZ => Some(AABB { min: new_pos(self.a_min, self.c - FUDGE, self.b_min), max: new_pos(self.a_max, self.c + FUDGE, self.b_max) }),
            AARectWhich::YZ => Some(AABB { min: new_pos(self.c - FUDGE, self.a_min, self.b_min), max: new_pos(self.c + FUDGE, self.a_max, self.b_max) }),
        }
    }
}

pub struct Clump {
    pub bounds: Option<AABB>,
    pub objects: Vec<Sphere>,
}

impl Clump {
    pub fn new(objects: Vec<Sphere>) -> Clump {
        let bounds = Self::compute_bounds(&objects);
        Clump { bounds, objects }
    }
    fn compute_bounds( objects: &[Sphere]) -> Option<AABB> {
        let mut iter = objects.iter();

        let first = match iter.next() {
            None => return None,
            Some(obj) => obj,
        };
        let mut bounds = match first.bounds() {
            None => return None,
            Some(aabb) => aabb,
        };

        for obj in iter {
            match obj.bounds() {
                None => return None,
                Some(aabb) => bounds.union_assign(&aabb),
            };
            
        }

        Some(bounds)
    }
}

impl Hitable for Clump {
    fn hit<'a>(&'a self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        // Bounds check for the clump
        // Note the full t range; otherwise the segment is inside completely
        if let Some(ref b) = self.bounds {
            if !b.hit(&ray, t_min, t_max) {
                return None;
            }
        }

        // Check each contained object
        let mut result = None;
        let mut closest_so_far = t_max;
        for obj in &self.objects {
            if let Some(record) = obj.hit(&ray, t_min, closest_so_far) {
                closest_so_far = record.t;
                result = Some(record);
            };
        }

        result
    }
    fn bounds(&self) -> Option<AABB> {
        match self.bounds {
            None => None,
            Some(ref b) => Some(AABB { min: b.min, max: b.max }),
        }
    }
}

/// A Node in a Bounding Volume Hierarchy.
/// This is a binary tree that ultimately contains a Hitable
pub enum BVH<'a> {
    Node { bounds: AABB, left: Box<BVH<'a>>, right: Box<BVH<'a>> },
    Leaf { bounds: AABB, object: &'a Box<dyn Hitable> },
}

impl<'a> BVH<'a> {
    fn compare_x_min(a: &Box<dyn Hitable>, b: &Box<dyn Hitable>) -> Ordering {
        let a_val: f32 = a.bounds().unwrap().min.x;
        let b_val: f32 = b.bounds().unwrap().min.x;
        a_val.partial_cmp(&b_val).unwrap()
    }
    fn _compare_y_min(a: &Box<dyn Hitable>, b: &Box<dyn Hitable>) -> Ordering {
        let a_val: f32 = a.bounds().unwrap().min.y;
        let b_val: f32 = b.bounds().unwrap().min.y;
        a_val.partial_cmp(&b_val).unwrap()
    }
    fn compare_z_min(a: &Box<dyn Hitable>, b: &Box<dyn Hitable>) -> Ordering {
        let a_val: f32 = a.bounds().unwrap().min.z;
        let b_val: f32 = b.bounds().unwrap().min.z;
        a_val.partial_cmp(&b_val).unwrap()
    }
    // TODO: Return Result<BVH,String>
    pub fn build<'b>(objects: &'b mut [Box<dyn Hitable>]) -> BVH {
        // Base case of recursion
        let num_objects = objects.len();
        if num_objects == 1 {
            let obj = &objects[0];
            let bounds = obj.bounds().expect("BVH can only hold objects with finite bounds");
            return BVH::Leaf { bounds, object: obj };
        }

        // Choose a random axis, sort objects
        // Note that we assume objects to be mainly spread around the XZ plane
        match rand::thread_rng().gen_range(0,2) {
            0 => objects.sort_unstable_by(Self::compare_x_min),
            1 => objects.sort_unstable_by(Self::compare_z_min),
            // 2 => objects.sort_unstable_by(Self::compare_y_min),
            _ => panic!("Unexpected random number encountered"),
        };

        // Partition the slice into two lists
        let (left,right) = objects.split_at_mut(num_objects / 2);
        let left = Self::build(left);
        let right = Self::build(right);
        let bounds = left.bounds().union(right.bounds());
        BVH::Node { bounds, left: Box::new(left), right: Box::new(right) }
    }

    pub fn glue(bvh: BVH<'a>, object: &'a Box<dyn Hitable>) -> BVH<'a> {
        let left = BVH::Leaf { bounds: object.bounds().unwrap(), object };
        let right = bvh;
        let bounds = left.bounds().union(right.bounds());
        BVH::Node { bounds, left: Box::new(left), right: Box::new(right) }
    }

    fn bounds(&self) -> &AABB {
        match self {
            BVH::Node{bounds, ..} => bounds,
            BVH::Leaf{bounds, ..} => bounds,
        }
    }

    pub fn hit(&'a self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord<'a>> {
        // Bounds check for the clump
        // Note the full t range; otherwise the segment is inside completely
        if !self.bounds().hit(&ray, t_min, t_max) {
            return None;
        };

        // Thunk to contained object for leaf, and extract subtrees for nodes
        let (left, right) = match self {
            BVH::Leaf{object, ..} => return object.hit(&ray, t_min, t_max),
            BVH::Node{ref left, ref right, ..} => (left, right),
        };

        // Check ray against each subtree
        let left_result = left.hit(&ray, t_min, t_max);
        let right_result = right.hit(&ray, t_min, t_max);

        // Tricky logic to return if either or both result is a miss
        let left_t = match left_result {
            None => return right_result,
            Some(ref r) => r.t,
        };
        let right_t = match right_result {
            None => return left_result,
            Some(ref r) => r.t,
        };
        // If both are hits then return the closest
        if left_t < right_t {
            left_result
        }
        else {
            right_result
        }
    }
}

pub fn hit<'a>(ray: &Ray, t_min: f32, t_max: f32, objects: &'a [Box<dyn Hitable>]) -> Option<HitRecord<'a>> {
    // // This algorithm seems like the more Rust-like way to do it.
    // // But because it doesn't get to prune future checks based on already seen objects, it is slower.
    // // Perhaps it would be better with multiple threads, or spatially grouped objects
    // objects.iter()
    //         .map(|obj| obj.hit(&ray, t_min, t_max))
    //         .filter_map(|h| h)
    //         .min_by(|h1, h2| h1.t.partial_cmp(&h2.t).unwrap())

    let mut result = None;
    let mut closest_so_far = t_max;
    for obj in objects {
        if let Some(record) = (*obj).hit(&ray, t_min, closest_so_far) {
            closest_so_far = record.t;
            result = Some(record);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::super::materials::Invisible;
    use super::*;

    #[test]
    fn hit_sphere_works() {
        let origin = new_pos(0.0, 0.0, 0.0);
        let left = new_pos(-2.0, 0.0, 0.0);
        let down_y = Ray { origin, direction: new_dir(0.0, -1.0, 0.0) };
        let down_y_parallel = Ray { origin: left, direction: new_dir(0.0, -1.0, 0.0) };
        // Expected hit: ray along y axis and sphere 2 units down y axis
        let sphere = Sphere { centre: new_pos(0.0, -2.0, 0.0), radius: 1.0, material: Box::new(Invisible {}) };
        match sphere.hit(&down_y, 0.0, 1000.0) {
            None => panic!("This ray and sphere were supposed to hit"),
            Some(record) => assert_eq!(record.t, 1.0),
        };
        // Expected miss: ray parallel to y axis and sphere 2 units down y axis
        match sphere.hit(&down_y_parallel, 0.0, 1000.0) {
            None => (),
            Some(_) => panic!("This ray and sphere were supposed to miss"),
        };
    }

    #[test]
    fn ray_aabb_hit() {
        // Given: An AABB
        let aabb = AABB { min: new_pos( -2.0, -2.0, -2.0), max: new_pos(2.0, -1.5, 2.0) };
        let origin = new_pos(0.0, 0.0, 0.0);
        let along_x = new_pos(3.0, 0.0, 0.0);
        let y_axis = new_dir(0.0, 1.0, 0.0);

        // Temp test: construct the ray separately to allow debugging
        let ray = Ray { origin: origin, direction: -y_axis };
        assert_eq!(aabb.hit(&ray, 0.0, 1000.0), true);

        // When: we check for a hit then each ray raturns appropriately
        assert_eq!(aabb.hit(&Ray { origin: origin,  direction: -y_axis }, 0.0, 1000.0), true);  // ray pointing into box
        assert_eq!(aabb.hit(&Ray { origin: origin,  direction: -y_axis }, 0.0,    1.0), false); // ray pointing into box but t range too short
        assert_eq!(aabb.hit(&Ray { origin: origin,  direction:  y_axis }, 0.0, 1000.0), false); // ray pointing away from box
        assert_eq!(aabb.hit(&Ray { origin: along_x, direction: -y_axis }, 0.0, 1000.0), false); // ray parallel to y axis and along x
        assert_eq!(aabb.hit(&Ray { origin: along_x, direction:  y_axis }, 0.0, 1000.0), false); // ray parallel to y axis and along x, pointing away
    }

}
