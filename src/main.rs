extern crate raytrace;
extern crate rand;

use std::error::Error;
use std::io::prelude::*;
use std::fs::File;
use std::path::Path;

use rand::Rng;

use raytrace::math::{Vec3,Ray};
use raytrace::ppm::PpmImage;
use raytrace::geometry;
use raytrace::geometry::*;
use raytrace::geometry::HitResult::{Hit,Miss};

struct Camera {
    lower_left : Vec3,
    horizontal : Vec3,
    vertical : Vec3,
    origin : Vec3,
}

impl Camera {
    fn new(aspect_ratio: f32) -> Camera {
        Camera {
            lower_left: Vec3::new(-1.0 * aspect_ratio, -1.0, -1.0),
            horizontal: Vec3::new(2.0 * aspect_ratio, 0.0, 0.0),
            vertical: Vec3::new(0.0, 2.0, 0.0),
            origin: Vec3::new(0.0, 0.0, 0.0)
        }
    }
    fn clip_to_ray(&self, u: f32, v: f32) -> Ray {
        Ray::new(&self.origin, &self.lower_left.add(&self.horizontal.mul(u)).add(&self.vertical.mul(v)))
    }
}

fn main() {
    const COLS: usize = 400;
    const ROWS: usize = 200;
    const NUM_SAMPLES: usize = 20; // Sample code recommends 100 but this is slow

    println!("Hello, world!");

    let camera = Camera::new(COLS as f32 / ROWS as f32);
    let mut objects : Vec<Box<Hitable>> = Vec::new();
    objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.0, -1.0), radius: 0.5 }));
    objects.push(Box::new(Sphere { centre: Vec3::new(0.0, -100.5, -1.0), radius: 100.0 }));

    let mut image = PpmImage::create(COLS, ROWS);
    let mut rng = rand::thread_rng();
    for r in (0..ROWS).rev() {
        let pv = r as f32;
        for c in 0..COLS {
            let pu = c as f32;
            // Anti-aliased: average colour from multiple randomised samples per pixel
            let mut colour = Vec3::new(0.0, 0.0, 0.0);
            for _s in 0..NUM_SAMPLES {
                let u = (pu + rng.gen::<f32>()) / COLS as f32;
                let v = (pv + rng.gen::<f32>()) / ROWS as f32;
                let ray = camera.clip_to_ray(u, v);
                colour = colour.add(&ray_colour(&ray, &objects));
            }
            let colour = colour.mul(1.0 / NUM_SAMPLES as f32);
            // Gamma correction: sqrt the colour
            let colour = Vec3::new(colour.x.sqrt(), colour.y.sqrt(), colour.z.sqrt());
            // Output the colour for this pixel
            image.append_pixel(&colour);
        }
    }

    // Output the image to a file
    let path = Path::new("out/output.ppm");
    write_text_to_file(&image.get_text(), &path);
}

fn random_in_unit_sphere() -> Vec3 {
    let unit = Vec3::new(1.0, 1.0, 1.0);
    let mut p = Vec3::new(2.0, 2.0, 2.0);
    let mut rng = rand::thread_rng();
    while p.len_sq() >= 1.0 {
        p = Vec3::new(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>()).mul(2.0).sub(&unit);
    }
    p
}

fn ray_colour(ray: &Ray, objects: &Vec<Box<Hitable>>) -> Vec3 {
    match geometry::hit(&ray, 0.001, 1000.0, &objects) {
        Hit { t: _, p, normal } => {
            // normal.normalise().add(&Vec3::new(1.0, 1.0, 1.0)).mul(0.5) // Use this return value to visualise normals
            let target = p.add(&normal).add(&random_in_unit_sphere());
            let dir = target.sub(&p);
            ray_colour(&Ray { origin: p, direction: dir }, &objects).mul(0.5)
        },
        Miss => {
            let unit = ray.direction.normalise();
            let t = 0.5 * (unit.y + 1.0);
            let blue = Vec3::new(0.5, 0.7, 1.0);
            let white = Vec3::new(1.0, 1.0, 1.0);
            white.mul(1.0 - t).add(&blue.mul(t))
        }
    }
}

fn write_text_to_file(text: &str, path: &Path) {
    let display = path.display();

    let mut file = match File::create(&path) {
        Err(why) => panic!("Failed to create file {}: {}", display, why.description()),
        Ok(file) => file,
    };

    match file.write_all(text.as_bytes()) {
        Err(why) => panic!("Failed to create file {}: {}", display, why.description()),
        Ok(_) => println!("Wrote to file {}!", display),
    }
}
