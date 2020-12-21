#![deny(bare_trait_objects)]
#![feature(stdsimd)]

extern crate rand;
extern crate noise;

pub mod math;
pub mod ppm;
pub mod geometry;
pub mod simdgeometry;
pub mod materials;

use math::*;
use geometry::BVH;

/// A camera that can cast a ray into the scene
pub struct Camera {
    lower_left : Vec3,
    horizontal : Vec3,
    vertical : Vec3,
    eye : Vec3,
    u: Vec3,
    v: Vec3,
    _w: Vec3,
    lens_radius: f32,
}

impl Camera {
    // Construct a new camera given a location & screen configuration
    pub fn new(eye: Vec3, lookat: Vec3, vup: Vec3, vfov: f32, aspect_ratio: f32, aperture: f32, focus_dist: f32) -> Camera {
        let lens_radius = 0.5 * aperture;

        // Compute Field of View
        let theta = vfov * std::f32::consts::PI / 180.0;
        let half_height = (0.5 * theta).tan();
        let half_width = aspect_ratio * half_height;

        // Compute basis
        let w = (eye - lookat).normalise();
        let u = cross(vup, w).normalise();
        let v = cross(w, u);

        // Compute vectors used to reconstitute the virtual screen that we project onto
        let lower_left = eye - u*(focus_dist*half_width) - v*(focus_dist*half_height) - w*focus_dist;
        let horizontal = u*(2.0 * focus_dist*half_width);
        let vertical = v*(2.0 * focus_dist*half_height);

        Camera { lower_left, horizontal, vertical, eye, u, v, _w: w, lens_radius }
    }

    /// Cast a ray into the scene from the given position in clip space
    pub fn clip_to_ray(&self, u: f32, v: f32) -> Ray {
        assert!(u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0, "Invalid clip space coordinates.  Expected values in [0,1]; found: [{},{}]", u, v);

        let mut eye = self.eye;
        if self.lens_radius > 0.0 {
            // Add noise to eye location to achieve depth-if-field
            // This chooses a random point on a disk sitting at the eye location, oriented to view direction
            let rd = Vec3::random_in_unit_disk(&mut rand::thread_rng()) * self.lens_radius;
            let offset = self.u * rd.x + self.v * rd.y;
            eye = self.eye + offset;
        }
        Ray::new(eye, self.lower_left + self.horizontal * u + self.vertical * v - eye)
    }
}

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
            // Vec3::new(0.0, 0.0, 0.0)
            let unit = ray.direction.normalise();
            let t = 0.5 * (unit.y + 1.0);
            let blue = Vec3::new(0.25, 0.35, 0.5);
            let white = Vec3::new(0.4, 0.4, 0.4);
            (white*(1.0 - t) + blue*t, 1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera() {
        // A specially crafted projection to simplify tests
        const ASPECT_RATIO: f32 = 1.5;
        const APERTURE: f32 = 0.0;
        const FIELD_OF_VIEW: f32 = 90.0;
        let eye = Vec3::new(1.0, 0.0, 0.0);
        let focus = Vec3::new(2.0, 0.0, 0.0);
        let up = Vec3::new(0.0, 1.0, 0.0);
        let dist_to_focus = (focus - eye).length();
        let camera = Camera::new(eye, focus, up, FIELD_OF_VIEW, ASPECT_RATIO, APERTURE, dist_to_focus);

        // Centre of the screen: ray straight down the X axis
        assert_eq!(camera.clip_to_ray(0.5, 0.5), Ray { origin: eye, direction: Vec3::new(1.0,  0.0,  0.0)});
        // Bottom left and top right of screen correspond to a rectangle based on aspect ratio
        assert_eq!(camera.clip_to_ray(0.0, 0.0), Ray { origin: eye, direction: Vec3::new(1.0, -1.0, -1.5)});
        assert_eq!(camera.clip_to_ray(1.0, 1.0), Ray { origin: eye, direction: Vec3::new(1.0,  1.0,  1.5)});
    }

    #[test]
    fn test_camera_invalid_clip() {
        const ASPECT_RATIO: f32 = 1.5;
        const APERTURE: f32 = 0.0;
        const FIELD_OF_VIEW: f32 = 90.0;
        let eye = Vec3::new(1.0, 0.0, 0.0);
        let focus = Vec3::new(2.0, 0.0, 0.0);
        let up = Vec3::new(0.0, 1.0, 0.0);
        let dist_to_focus = (focus - eye).length();
        let camera = Camera::new(eye, focus, up, FIELD_OF_VIEW, ASPECT_RATIO, APERTURE, dist_to_focus);

        // Should panic due to invalid clip coords (ideally I'd check the message, but haven't dug out the type of catch_unwind yet)
        assert!(std::panic::catch_unwind(|| camera.clip_to_ray( 1.5,  0.2)).is_err());
        assert!(std::panic::catch_unwind(|| camera.clip_to_ray( 0.9, -0.2)).is_err());
        assert!(std::panic::catch_unwind(|| camera.clip_to_ray(-0.9,  0.2)).is_err());
    }
}
