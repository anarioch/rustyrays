#![deny(bare_trait_objects)]
#![feature(stdsimd)]

extern crate rand;
extern crate noise;
extern crate serde;

pub mod math;
pub mod ppm;
pub mod geometry;
pub mod simdgeometry;
pub mod materials;
pub mod camera;
pub mod scene;

use math::*;
use geometry::BVH;

/// Cast a ray into the scene represented by the spatial lookup, returning a colour
/// Depth should decrease by one for each bounced ray, terminating recursion onces it reaches zero
/// Returns the colour and number of rays cast
pub fn cast_ray(ray: &Ray, object: &BVH, depth: usize) -> (Vec3, usize) {
    match object.hit(&ray, 0.001, 1000.0) {
        Some(record) => {
            // (normal.normalise() + Vec3::new(1.0, 1.0, 1.0)) * 0.5 // Use this return value to visualise normals
            if depth == 0 {
                return (Vec3::new(0.0, 0.0, 0.0), 1);
            }
            let emission = materials::emit(record.material);
            match materials::scatter(&ray, &record) {
                Some(scatter) => {
                    let (recurse,count) = cast_ray(&scatter.scattered, &object, depth - 1);
                    (emission + scatter.attenuation.mul_vec(recurse), count + 1)
                },
                None => (emission, 1)
            }
        },
        None => {
            (Vec3::splat(0.0), 1)
            // let unit = ray.direction.normalise();
            // let t = 0.5 * (unit.y + 1.0);
            // let blue = Vec3::new(0.25, 0.35, 0.5);
            // let white = Vec3::new(0.4, 0.4, 0.4);
            // (white*(1.0 - t) + blue*t, 1)
        }
    }
}
