#![deny(bare_trait_objects)]

extern crate raytrace;
extern crate rand;
extern crate serde;


use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;

use rand::Rng;

use serde::{Deserialize, Serialize};

use raytrace::math::*;
use raytrace::ppm::PpmImage;
use raytrace::geometry::*;
use raytrace::materials::*;
use raytrace::Camera;

/// Definition of a scene to be rendered, including objects and camera.
/// Objects are split into two groups, to provide hints for BVH construction
struct Scene {
    objects: Vec<Box<dyn Hitable>>,
    outlier_objects: Vec<Box<dyn Hitable>>,
    camera: Camera,
}

fn random_scene(aspect_ratio: f32) -> Scene {
    let mut objects : Vec<Box<dyn Hitable>> = Vec::new();
    let mut outlier_objects : Vec<Box<dyn Hitable>> = Vec::new();
    let mut rng = rand::thread_rng();
    let mut rand = || rng.gen::<f32>();

    let world_centre = Vec3::new(0.0, -1000.0, 0.0);
    let world_radius = 1000.0;

    // Create closure that creates a randomised sphere within the x,z unit cell
    let rad_sq = world_radius * world_radius;
    let mut random_sphere = |x, z| {
        let radius = 0.2;
        let mut centre = Vec3::new(x + 0.9 * rand(), 0.0, z + 0.9 * rand());
        centre.y = (rad_sq - centre.x * centre.x).sqrt() - world_radius + radius;
        let material: Material = match rand() {
            d if d < 0.65 => Material::Lambertian { albedo: Vec3::new(rand() * rand(), rand() * rand(), rand() * rand()) },
            d if d < 0.85 => Material::Metal { albedo: Vec3::new(0.5 * (1.0 + rand()), 0.5 * (1.0 + rand()), 0.5 * (1.0 + rand())), fuzz: 0.5 * rand() },
            _ => Material::Dielectric { ref_index: 1.5 },
        };
        Sphere { centre, radius, material }
    };

    for a in -7..7 {
        for b in -7..7 {
            objects.push(Box::new(random_sphere(a as f32, b as f32)));
        }
    }

    // Three feature spheres, showing off some of the materials
    // let reddish = Material::Lambertian { albedo: Box::new(ConstantTexture { colour: Vec3::new(0.7, 0.2, 0.3) }) };
    let gold = Material::Metal { albedo: Vec3::new(0.8, 0.6, 0.2), fuzz: 0.0 };
    let marble = Material::PolishedStone { albedo: Box::new(NoiseTexture::new(12.0, Vec3::new(0.6, 0.1, 0.2))) };
    let glass = Material::Dielectric { ref_index: 1.5 };
    objects.push(Box::new(Sphere { centre: Vec3::new(-4.0, 0.5, -1.0), radius: 0.5, material: gold }));
    objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.5, -1.0), radius: 0.5, material: glass }));
    objects.push(Box::new(Sphere { centre: Vec3::new(4.0, 0.5, -1.0), radius: 0.5, material: marble }));

    // A glowing orb up above all other objects to light the scene
    let bulb = Material::DiffuseLight { emission_colour: Vec3::new(2.0, 2.0, 2.0) };
    outlier_objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 10.0, -1.0), radius: 5.0, material: bulb }));

    // Neat trick: embed a small sphere in another to simulate glass.  Might work by reversing coefficient also
    // let glass2 = Material::Dielectric { ref_index: 1.5 };
    // objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.0, -1.0), radius: 0.5, material: glass }));
    // objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.0, -1.0), radius: -0.45, material: glass2 }));

    // A gold wall
    let brushed_gold = Material::Metal { albedo: Vec3::new(1.0, 0.85, 0.0), fuzz: 0.3 };
    // let brushed_copper = Material::Metal { albedo: Vec3::new(0.7, 0.45, 0.2), fuzz: 0.3 };
    outlier_objects.push(Box::new(AARect { which: AARectWhich::XY, a_min: -4.0, a_max: 4.0, b_min: -4.0, b_max: 1.3, c: -1.7, negate_normal: false, material: brushed_gold}));

    // The giant world sphere on which all others sit
    let globe_texture = CheckerTexture {
        check_size: 10.0,
        odd: Vec3::new(0.2, 0.3, 0.1),
        even: Vec3::new(0.9, 0.9, 0.9),
    };
    outlier_objects.push(Box::new(Sphere {
        centre: world_centre,
        radius: world_radius,
        material: Material::TexturedLambertian { albedo: Box::new(globe_texture) },
    }));

    // Configure the camera
    const APERTURE: f32 = 0.03;
    const FIELD_OF_VIEW: f32 = 20.0;
    let eye = Vec3::new(10.0, 1.1, 0.3);
    let focus = Vec3::new(4.0, 0.55, -0.3);
    let up = Vec3::new(0.0, 1.0, 0.0);
    let dist_to_focus = (focus - eye).length();
    let camera = Camera::new(eye, focus, up, FIELD_OF_VIEW, aspect_ratio, APERTURE, dist_to_focus);

    Scene { objects, outlier_objects, camera }
}

fn noise_scene(aspect_ratio: f32) -> Scene {
    let mut objects : Vec<Box<dyn Hitable>> = Vec::new();
    let mut outlier_objects : Vec<Box<dyn Hitable>> = Vec::new();

    // The giant world sphere on which all others sit
    let noise1 = Material::TexturedLambertian { albedo: Box::new(NoiseTexture::new(4.0, Vec3::new(1.0, 1.0, 1.0))) };
    outlier_objects.push(Box::new(Sphere {
        centre: Vec3::new(0.0, -1000.5, -2.0),
        radius: 1000.0,
        material: noise1,
    }));

    let marble = Material::PolishedStone { albedo: Box::new(NoiseTexture::new(12.0, Vec3::new(0.6, 0.2, 0.1))) };
    objects.push(Box::new(Sphere { centre: Vec3::new(4.0, 0.0, -1.0), radius: 0.5, material: marble }));

    // Configure the camera
    const APERTURE: f32 = 0.03;
    const FIELD_OF_VIEW: f32 = 20.0;
    let eye = Vec3::new(10.0, 1.1, 0.3);
    let focus = Vec3::new(4.0, 0.55, -0.3);
    let up = Vec3::new(0.0, 1.0, 0.0);
    let dist_to_focus = (focus - eye).length();
    let camera = Camera::new(eye, focus, up, FIELD_OF_VIEW, aspect_ratio, APERTURE, dist_to_focus);

    Scene { objects, outlier_objects, camera }
}

fn clamp(mut x: f32, min: f32, max: f32) -> f32 {
    if x < min { x = min; }
    if x > max { x = max; }
    x
}

#[derive(Debug, Deserialize, Serialize)]
struct TargetParameters {
    cols: usize,
    rows: usize,
    samples_per_pixel: usize,
    max_bounces: usize
}

fn read_params_from_file<P: AsRef<Path>>(path: P) -> Result<TargetParameters, Box<dyn Error>> {
    // Open the file in read-only mode with buffer.
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `TargetParameters`.
    let params = serde_json::from_reader(reader)?;

    // Return the `TargetParameters`.
    Ok(params)
}

fn main() {
    let params: TargetParameters = read_params_from_file("params.json").unwrap();
    let aspect_ratio: f32 = params.cols as f32 / params.rows as f32;

    println!("Welcome to JDs Rustaceous Raytracer!");

    // Set up variables for timing and progress printing
    let max_iterations = params.cols * params.rows;
    let mut num_iterations = 0;
    let start_time = std::time::Instant::now();
    let mut last_time = std::time::Instant::now();
    let io_flush = || std::io::stdout().flush().expect("Could not flush stdout");
    io_flush();
    let mut total_rays = 0;

    // Generate the scene
    let scene_index = 1;
    let mut scene = match scene_index {
        0 => noise_scene(aspect_ratio),
        _ => random_scene(aspect_ratio),
    };

    // Split outliers from rest of objects here to improve AABB sizes
    // TODO Figure out better heuristics to this automatically
    let obj_slice = &mut scene.objects;
    let mut bvh = BVH::build(obj_slice);
    for ref outlier in &scene.outlier_objects {
        bvh = BVH::glue(bvh, outlier);
    }
    let bvh = bvh;

    println!("Scene generation done (Took {:.2}s)", start_time.elapsed().as_secs());
    let post_scene_gen_time = std::time::Instant::now();

    // Cast rays to generate the image
    let mut rng = rand::thread_rng();
    let mut img : Vec<Vec3> = Vec::with_capacity(params.cols * params.rows);
    img.resize(params.cols * params.rows, Vec3::splat(0.0));
    for s in 0..params.samples_per_pixel {
        let mut image = PpmImage::create(params.cols, params.rows);
        for r in (0..params.rows).rev() {
            let pv = r as f32;
            for c in 0..params.cols {
                let pu = c as f32;
                // Anti-aliased: average colour from multiple randomised samples per pixel
                let u = (pu + rng.gen::<f32>()) / params.cols as f32;
                let v = (pv + rng.gen::<f32>()) / params.rows as f32;
                let ray = scene.camera.clip_to_ray(u, v);
                let (ray_colour, ray_count) = raytrace::cast_ray(&ray, &bvh, params.max_bounces);
                total_rays += ray_count;

                // Add the colour to our accumulator
                let mut colour = img[params.cols * r + c] + ray_colour;
                img[params.cols * r + c] = colour;

                colour *= 1.0 / (s + 1) as f32;
                // Clamp the colour to [0..1]
                let colour = colour.map(|x| clamp(x, 0.0, 1.0));
                // Gamma correction: sqrt the colour
                let colour = colour.map(|x| x.sqrt());
                // Output the colour for this pixel
                image.append_pixel(colour);

                num_iterations += 1;
            }

            if last_time.elapsed().as_secs() >= 1 {
                let percentage_done = 100.0 * num_iterations as f32 / max_iterations as f32 / params.samples_per_pixel as f32;
                let time_elapsed = post_scene_gen_time.elapsed();
                let time_elapsed = 1_000_000_000.0 * time_elapsed.as_secs() as f32 + time_elapsed.subsec_nanos() as f32;
                let time_per_ray = time_elapsed / total_rays as f32;
                print!("\rProcessed {:2.2}%;  {} bounces so far;  {:3.2}ns/ray                    ", percentage_done, total_rays, time_per_ray);
                io_flush();
                last_time += std::time::Duration::from_secs(1);
            }
        }

        // Output the image to a file
        let path = Path::new("out/output.ppm");
        write_text_to_file(&image.get_text(), &path, false);
    }

    let time_elapsed = post_scene_gen_time.elapsed();
    let time_elapsed = 1_000_000_000.0 * time_elapsed.as_secs() as f32 + time_elapsed.subsec_nanos() as f32;
    let time_per_ray = time_elapsed / total_rays as f32;
    println!("\rProcessing done.                                                ");
    println!("    Rows:       {}", params.rows);
    println!("    Columns:    {}", params.cols);
    println!("    Rays/pixel: {}", params.samples_per_pixel);
    println!("    ns/ray:     {:.1}", time_per_ray);
    println!("    Time taken: {:.3}", time_elapsed / 1_000_000_000.0);
    println!("    Rays cast:  {}", total_rays);
    println!("Writing file..");
}

fn write_text_to_file(text: &str, path: &Path, write_status: bool) {
    let display = path.display();

    let mut file = match File::create(&path) {
        Err(why) => panic!("Failed to create file {}: {}", display, why),
        Ok(file) => file,
    };

    match file.write_all(text.as_bytes()) {
        Err(why) => panic!("Failed to write to file {}: {}", display, why),
        Ok(_) => if write_status {println!("Wrote to file {}!", display) },
    }
}
