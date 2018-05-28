extern crate raytrace;

use std::error::Error;
use std::io::prelude::*;
use std::fs::File;
use std::path::Path;

use raytrace::math;
use raytrace::math::{Vec3,Ray};
use raytrace::math::SphereHitResult::{Hit,Miss};
use raytrace::ppm::PpmImage;

fn main() {
    const COLS: usize = 500;
    const ROWS: usize = 300;

    println!("Hello, world!");

    let mut image = PpmImage::create(COLS, ROWS);
    let origin = Vec3::new(0.0, 0.0, 0.0);
    for r in 0..ROWS {
        let y = (ROWS-r-1) as f32 / ROWS as f32;
        let y = 2.0 * y - 1.0;
        for c in 0..COLS {
            let x = c as f32 / ROWS as f32;
            let x = 2.0 * x - (COLS as f32 / ROWS as f32);
            let ray = Ray::new(&origin, &Vec3::new(x, y, -1.0));
            let colour = colour(&ray);
            image.append_pixel(&colour);
        }
    }

    // Output the image to a file
    let path = Path::new("out/output.ppm");
    write_text_to_file(&image.get_text(), &path);
}

fn colour(ray: &Ray) -> Vec3 {
    let sphere_centre = Vec3::new(0.0, 0.0, -1.0);
    match math::hit_sphere(&sphere_centre, 0.5, &ray) {
        Hit { t } => {
            let normal = ray.at_t(t).sub(&sphere_centre).normalise();
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
    // let colour = if ray.direction.y > 0.0 {
    //     Vec3::new(0.5, 0.7, 0.95)
    // }
    // else {
    //     Vec3::new(0.4, 0.8, 0.5)
    // };
    // colour
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
