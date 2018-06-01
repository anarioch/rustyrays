extern crate raytrace;
extern crate rand;

use std::error::Error;
use std::io::prelude::*;
use std::fs::File;
use std::path::Path;

use rand::Rng;

use raytrace::math;
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
    fn new(lookfrom: Vec3, lookat: &Vec3, vup: &Vec3, vfov: f32, aspect_ratio: f32) -> Camera {
        // Compute Field of View
        let theta = vfov * std::f32::consts::PI / 180.0;
        let half_height = (0.5 * theta).tan();
        let half_width = aspect_ratio * half_height;

        // Compute basis
        let w = lookfrom.sub(&lookat).normalise();
        let u = vup.cross(&w).normalise();
        let v = w.cross(&u);

        // // Diagnostics
        // println!("u = {:?}", u);
        // println!("v = {:?}", v);
        // println!("w = {:?}", w);

        let lower_left = lookfrom.sub(&u.mul(half_width)).sub(&v.mul(half_height)).sub(&w);
        let horizontal = u.mul(2.0 * half_width);
        let vertical = v.mul(2.0 * half_height);
        let origin = lookfrom;

        // // Diagnostics
        // println!("origin     = {:?}", origin);
        // println!("lower_left = {:?}", lower_left);
        // println!("horizontal = {:?}", horizontal);
        // println!("vertical   = {:?}", vertical);

        Camera { lower_left, horizontal, vertical, origin }
    }

    fn clip_to_ray(&self, u: f32, v: f32) -> Ray {
        Ray::new(&self.origin, &self.lower_left.add(&self.horizontal.mul(u)).add(&self.vertical.mul(v)))
    }
}

fn random_scene() -> Vec<Box<Hitable>> {
    let mut objects : Vec<Box<Hitable>> = Vec::new();
    let mut rng = rand::thread_rng();

    // The giant world sphere on which all others sit
    objects.push(Box::new(Sphere {
        centre: Vec3::new(0.0, -1000.5, -2.0),
        radius: 1000.0,
        material: Box::new(Lambertian { albedo: Vec3::new(0.5, 0.5, 0.5) }),
    }));

    // Randomise a bunch of spheres
    for a in -5..5 {
        for b in -5..5 {
            let centre = Vec3::new(a as f32 + 0.9 * rng.gen::<f32>(), -0.3, b as f32 + 0.9 * rng.gen::<f32>());
            let material = Box::new(Metal { albedo: Vec3::new(0.5 * (1.0 + rng.gen::<f32>()), 0.5 * (1.0 + rng.gen::<f32>()), 0.5 * (1.0 + rng.gen::<f32>())), fuzz: 0.5 * rng.gen::<f32>() });
            objects.push(Box::new(Sphere { centre, radius: 0.2, material }));
        }
    }
    let reddish = Box::new(Lambertian { albedo: Vec3::new(0.7, 0.2, 0.3) });
    let brushed_gold = Box::new(Metal { albedo: Vec3::new(0.8, 0.6, 0.2), fuzz: 0.3 });
    // let silver = Box::new(Metal { albedo: Vec3::new(0.8, 0.8, 0.8), fuzz: 0.05 });
    let glass = Box::new(Dielectric { ref_index: 1.5 });
    let glass2 = Box::new(Dielectric { ref_index: 1.5 });
    objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.0, -2.0), radius: 0.5, material: reddish }));
    objects.push(Box::new(Sphere { centre: Vec3::new(1.0, 0.0, -2.0), radius: 0.5, material: brushed_gold }));
    objects.push(Box::new(Sphere { centre: Vec3::new(-1.0, 0.0, -2.0), radius: 0.5, material: glass }));
    objects.push(Box::new(Sphere { centre: Vec3::new(-1.0, 0.0, -2.0), radius: -0.45, material: glass2 }));

    objects
}

fn main() {
    const COLS: usize = 400;
    const ROWS: usize = 200;
    const NUM_SAMPLES: usize = 100; // Sample code recommends 100 but this is slow
    const MAX_BOUNCES: usize = 20;

    println!("Hello, world!");

    const ASPECT_RATIO: f32 = COLS as f32 / ROWS as f32;
    // Note: The projection is not working quite right, and distorts as the camera's origin moves away from the origin
    // let camera = Camera::new(Vec3::new(-2.0, 2.0, 1.0), &Vec3::new(0.0, 0.0, -1.0), &Vec3::new(0.0, 1.0, 0.0), 90.0, ASPECT_RATIO);
    let camera = Camera::new(Vec3::new(0.0, 0.0, 0.0), &Vec3::new(0.0, 0.0, -1.0), &Vec3::new(0.0, 1.0, 0.0), 45.0, ASPECT_RATIO);

    println!("Ray(0  ,0.5) = {:?}", &camera.clip_to_ray(0.0, 0.5));
    println!("Ray(0.5,0.5) = {:?}", &camera.clip_to_ray(0.5, 0.5));
    println!("Ray(1  ,0.5) = {:?}", &camera.clip_to_ray(1.0, 0.5));

    let objects = random_scene();
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
                colour.add_eq(&ray_colour(&ray, &objects, MAX_BOUNCES));
            }
            colour.mul_eq(1.0 / NUM_SAMPLES as f32);
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
        p.set(rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>());
        p.mul_eq(2.0);
        p.sub_eq(&unit);
    }
    p
}

struct Lambertian {
    albedo: Vec3,
}

impl Material for Lambertian {
    fn scatter(&self, _ray: &Ray, hit: &HitRecord) -> Option<ScatterResult> {
        let target = hit.p.add(&hit.normal).add(&random_in_unit_sphere());
        let dir = target.sub(&hit.p);
        let scattered = Ray { origin: hit.p.clone(), direction: dir };
        Some(ScatterResult { attenuation: self.albedo.clone(), scattered})
    }
}

struct Metal {
    albedo: Vec3,
    fuzz: f32,
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<ScatterResult> {
        let reflected = math::reflect(&ray.direction.normalise(), &hit.normal);
        let reflected = reflected.add(&random_in_unit_sphere().mul(self.fuzz));
        if reflected.dot(&hit.normal) > 0.0 {
            let scattered = Ray { origin: hit.p.clone(), direction: reflected };
            Some(ScatterResult { attenuation: self.albedo.clone(), scattered})
        }
        else {
            None
        }
    }
}

struct Dielectric {
    ref_index: f32,
}

fn schlick(cosine: f32, ref_index: f32) -> f32 {
    let r0 = (1.0 - ref_index) / (1.0 + ref_index);
    let r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<ScatterResult> {
        let dir = ray.direction.normalise();
        let reflected = math::reflect(&dir, &hit.normal);
        let attenuation = Vec3::new(1.0, 1.0, 1.0);

        let ray_dot_norm = ray.direction.dot(&hit.normal);
        let cosine = ray_dot_norm / ray.direction.len_sq().sqrt();
        let (outward_normal, ni_over_nt, cosine) =
            if ray_dot_norm > 0.0 {
                (hit.normal.mul(-1.0), self.ref_index, self.ref_index * cosine)
            }
            else {
                (hit.normal.clone(), 1.0 / self.ref_index, -cosine)
            };
        
        let (refracted, reflect_prob) =
            match math::refract(&dir, &outward_normal, ni_over_nt) {
                Some(refracted) => {
                    (refracted, schlick(cosine, self.ref_index))
                },
                None => (Vec3::new(0.0, 0.0, 0.0), 1.0)
            };
        let ray_dir = 
            if rand::thread_rng().gen::<f32>() < reflect_prob {
                reflected
            }
            else {
                refracted
            };
        let scattered = Ray { origin: hit.p.clone(), direction: ray_dir };
        Some(ScatterResult { attenuation, scattered })
    }
}

fn ray_colour(ray: &Ray, objects: &Vec<Box<Hitable>>, depth: usize) -> Vec3 {
    match geometry::hit(&ray, 0.001, 1000.0, &objects) {
        Hit(record) => {
            // normal.normalise().add(&Vec3::new(1.0, 1.0, 1.0)).mul(0.5) // Use this return value to visualise normals
            if depth == 0 {
                return Vec3::new(0.0, 0.0, 0.0);
            }
            match record.material.scatter(&ray, &record) {
                Some(scatter) => {
                    let recurse = ray_colour(&scatter.scattered, &objects, depth - 1);
                    scatter.attenuation.mul_vec(&recurse)
                },
                None => Vec3::new(0.0, 0.0, 0.0)
            }
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
