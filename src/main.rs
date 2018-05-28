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

fn main() {
    const COLS: usize = 500;
    const ROWS: usize = 300;
    const NUM_SAMPLES: usize = 100;

    println!("Hello, world!");

    let mut image = PpmImage::create(COLS, ROWS);
    let mut objects : Vec<Box<Hitable>> = Vec::new();
    objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.0, -1.0), radius: 0.5 }));
    objects.push(Box::new(Sphere { centre: Vec3::new(0.0, -100.5, -1.0), radius: 100.0 }));
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let mut rng = rand::thread_rng();
    for r in 0..ROWS {
        for c in 0..COLS {
            let mut colour = Vec3::new(0.0, 0.0, 0.0);
            for _s in 0..NUM_SAMPLES {
                let y = ((ROWS-r-1) as f32 + rng.gen::<f32>()) / ROWS as f32;
                let y = 2.0 * y - 1.0;
                let x = (c as f32 + rand::random::<f32>()) / ROWS as f32;
                let x = 2.0 * x - (COLS as f32 / ROWS as f32);
                let ray = Ray::new(&origin, &Vec3::new(x, y, -1.0));
                colour = colour.add(&ray_colour(&ray, &objects));
            }
            let colour = colour.mul(1.0 / NUM_SAMPLES as f32);
            image.append_pixel(&colour);
        }
    }

    // Output the image to a file
    let path = Path::new("out/output.ppm");
    write_text_to_file(&image.get_text(), &path);
}

fn ray_colour(ray: &Ray, objects: &Vec<Box<Hitable>>) -> Vec3 {
    match geometry::hit(&ray, 0.0, 1000.0, &objects) {
        Hit { t: _, p: _, normal } => {
            let normal = normal.normalise();
            Vec3::new(normal.x + 1.0, normal.y + 1.0, normal.z + 1.0).mul(0.5)
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
