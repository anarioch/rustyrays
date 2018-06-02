extern crate raytrace;
extern crate rand;

use std::error::Error;
use std::io::prelude::*;
use std::fs::File;
use std::path::Path;

use rand::Rng;

use raytrace::math;
use raytrace::math::*;
use raytrace::ppm::PpmImage;
use raytrace::geometry;
use raytrace::geometry::*;
use raytrace::geometry::HitResult::{Hit,Miss};

struct Camera {
    lower_left : Vec3,
    horizontal : Vec3,
    vertical : Vec3,
    eye : Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,
    lens_radius: f32,
}

impl Camera {
    fn new(eye: Vec3, lookat: &Vec3, vup: &Vec3, vfov: f32, aspect_ratio: f32, aperture: f32, focus_dist: f32) -> Camera {
        let lens_radius = 0.5 * aperture;

        // Compute Field of View
        let theta = vfov * std::f32::consts::PI / 180.0;
        let half_height = (0.5 * theta).tan();
        let half_width = aspect_ratio * half_height;

        // Compute basis
        let w = eye.sub(&lookat).normalise();
        let u = cross(&vup, &w).normalise();
        let v = cross(&w, &u);

        // Compute vectors used to reconstitute the virtual screen that we project onto
        let lower_left = eye.sub(&u.mul(focus_dist*half_width)).sub(&v.mul(focus_dist*half_height)).sub(&w.mul(focus_dist));
        let horizontal = u.mul(2.0 * focus_dist*half_width);
        let vertical = v.mul(2.0 * focus_dist*half_height);

        Camera { lower_left, horizontal, vertical, eye, u, v, w, lens_radius }
    }

    fn clip_to_ray(&self, u: f32, v: f32) -> Ray {
        let rd = random_in_unit_disk().mul(self.lens_radius);
        let offset = self.u.mul(rd.x).add(&self.v.mul(rd.y));
        let eye = self.eye.add(&offset);
        Ray::new(&eye, &self.lower_left.add(&self.horizontal.mul(u)).add(&self.vertical.mul(v)).sub(&eye))
    }
}

fn random_scene() -> Vec<Box<Hitable>> {
    let mut objects : Vec<Box<Hitable>> = Vec::new();
    let mut rng = rand::thread_rng();
    let mut rand = || rng.gen::<f32>();

    // The giant world sphere on which all others sit
    objects.push(Box::new(Sphere {
        centre: Vec3::new(0.0, -1000.5, -2.0),
        radius: 1000.0,
        material: Box::new(Lambertian { albedo: Vec3::new(0.5, 0.5, 0.5) }),
    }));

    // Create closure that creates a randomised sphere within the x,z unit cell
    let mut random_sphere = |x, z| {
        let centre = Vec3::new(x + 0.9 * rand(), -0.3, z + 0.9 * rand());
        let material: Box<Material> = match rand() {
            d if d < 0.65 => Box::new(Lambertian { albedo: Vec3::new(rand() * rand(), rand() * rand(), rand() * rand()) }),
            d if d < 0.85 => Box::new(Metal { albedo: Vec3::new(0.5 * (1.0 + rand()), 0.5 * (1.0 + rand()), 0.5 * (1.0 + rand())), fuzz: 0.5 * rand() }),
            _ => Box::new(Dielectric { ref_index: 1.5 }),
        };
        Sphere { centre, radius: 0.2, material }
    };

    let check_spheres = |clump: &Clump| {
        for sphere in clump.objects.iter() {
            let dist = sphere.centre.sub(&clump.bounds.centre).len_sq().sqrt();
            if dist + sphere.radius > clump.bounds.radius {
                panic!("Sphere is outside!");
            }
        }
    };
    // Randomise a bunch of spheres, putting them into a quadrant of clumps to optimise ray lookup
    let mut clump_a = Clump { bounds: Bounds { centre: Vec3::new(-3.5, -0.3, -3.5), radius: 1.5 * 3.5}, objects: Vec::new() };
    let mut clump_b = Clump { bounds: Bounds { centre: Vec3::new(-3.5, -0.3, 3.5), radius: 1.5 * 3.5}, objects: Vec::new() };
    let mut clump_c = Clump { bounds: Bounds { centre: Vec3::new(3.5, -0.3, -3.5), radius: 1.5 * 32.5}, objects: Vec::new() };
    let mut clump_d = Clump { bounds: Bounds { centre: Vec3::new(3.5, -0.3, 3.5), radius: 1.5 * 3.5}, objects: Vec::new() };
    for a in -7..0 { for b in -7..0 {
        clump_a.objects.push(random_sphere(a as f32, b as f32));
    } }
    for a in -7..0 { for b in 0..7 {
        clump_b.objects.push(random_sphere(a as f32, b as f32));
    } }
    for a in 0..7 { for b in -7..0 {
        clump_c.objects.push(random_sphere(a as f32, b as f32));
    } }
    for a in 0..7 { for b in 0..7 {
        clump_d.objects.push(random_sphere(a as f32, b as f32));
    } }
    check_spheres(&clump_a);
    check_spheres(&clump_b);
    check_spheres(&clump_c);
    check_spheres(&clump_d);
    objects.push(Box::new(clump_a));
    objects.push(Box::new(clump_b));
    objects.push(Box::new(clump_c));
    objects.push(Box::new(clump_d));

    let reddish = Box::new(Lambertian { albedo: Vec3::new(0.7, 0.2, 0.3) });
    // let brushed_gold = Box::new(Metal { albedo: Vec3::new(0.8, 0.6, 0.2), fuzz: 0.3 });
    let gold = Box::new(Metal { albedo: Vec3::new(0.8, 0.6, 0.2), fuzz: 0.0 });
    let glass = Box::new(Dielectric { ref_index: 1.5 });
    objects.push(Box::new(Sphere { centre: Vec3::new(-4.0, 0.0, -1.0), radius: 0.5, material: reddish }));
    objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.0, -1.0), radius: 0.5, material: glass }));
    objects.push(Box::new(Sphere { centre: Vec3::new(4.0, 0.0, -1.0), radius: 0.5, material: gold }));

    // Neat trick: embed a small sphere in another to simulate glass.  Might work by reversing coefficient also
    // let glass2 = Box::new(Dielectric { ref_index: 1.5 });
    // objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.0, -1.0), radius: 0.5, material: glass }));
    // objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.0, -1.0), radius: -0.45, material: glass2 }));

    objects
}

fn main() {
    const COLS: usize = 400;
    const ROWS: usize = 400;
    const NUM_SAMPLES: usize = 100; // Sample code recommends 100 but this is slow
    const MAX_BOUNCES: usize = 30;

    println!("Hello, world!");

    const ASPECT_RATIO: f32 = COLS as f32 / ROWS as f32;
    const APERTURE: f32 = 0.03;
    const FIELD_OF_VIEW: f32 = 20.0;
    let eye = Vec3::new(10.0, 0.5, 0.3);
    let focus = Vec3::new(4.0, 0.2, -0.3);
    let up = Vec3::new(0.0, 1.0, 0.0);
    let dist_to_focus = focus.sub(&eye).len_sq().sqrt();
    let camera = Camera::new(eye, &focus, &up, FIELD_OF_VIEW, ASPECT_RATIO, APERTURE, dist_to_focus);

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
    let mut p = Vec3::new(2.0, 2.0, 2.0);
    let mut rng = rand::thread_rng();
    let mut rand = || rng.gen::<f32>();
    while p.len_sq() >= 1.0 {
        p.set(2.0 * rand() - 1.0, 2.0 * rand() - 1.0, 2.0 * rand() - 1.0);
    }
    p
}

fn random_in_unit_disk() -> Vec3 {
    let mut p = Vec3::new(2.0, 2.0, 2.0);
    let mut rng = rand::thread_rng();
    let mut rand = || rng.gen::<f32>();
    while p.len_sq() >= 1.0 {
        p.set(2.0 * rand() - 1.0, 2.0 * rand() - 1.0, 0.0);
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
        if dot(&reflected, &hit.normal) > 0.0 {
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

        let ray_dot_norm = dot(&ray.direction, &hit.normal);
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
