#![deny(bare_trait_objects)]

extern crate raytrace;
extern crate rand;

use std::error::Error;
use std::io::prelude::*;
use std::fs::File;
use std::path::Path;

use rand::Rng;

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
        let material: Box<dyn Material> = match rand() {
            d if d < 0.65 => Box::new(Lambertian { albedo: Vec3::new(rand() * rand(), rand() * rand(), rand() * rand()) }),
            d if d < 0.85 => Box::new(Metal { albedo: Vec3::new(0.5 * (1.0 + rand()), 0.5 * (1.0 + rand()), 0.5 * (1.0 + rand())), fuzz: 0.5 * rand() }),
            _ => Box::new(Dielectric { ref_index: 1.5 }),
        };
        Sphere { centre, radius, material }
    };

    for a in -7..7 {
        for b in -7..7 {
            objects.push(Box::new(random_sphere(a as f32, b as f32)));
        }
    }

    // Three feature spheres, showing off some of the materials
    // let reddish = Box::new(Lambertian { albedo: Box::new(ConstantTexture { colour: Vec3::new(0.7, 0.2, 0.3) }) });
    let gold = Box::new(Metal { albedo: Vec3::new(0.8, 0.6, 0.2), fuzz: 0.0 });
    let marble = Box::new(PolishedStone { albedo: Box::new(NoiseTexture::new(12.0, Vec3::new(0.6, 0.1, 0.2))) });
    let glass = Box::new(Dielectric { ref_index: 1.5 });
    objects.push(Box::new(Sphere { centre: Vec3::new(-4.0, 0.5, -1.0), radius: 0.5, material: gold }));
    objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.5, -1.0), radius: 0.5, material: glass }));
    objects.push(Box::new(Sphere { centre: Vec3::new(4.0, 0.5, -1.0), radius: 0.5, material: marble }));

    // A glowing orb up above all other objects to light the scene
    let bulb = Box::new(DiffuseLight { emission_colour: Vec3::new(2.0, 2.0, 2.0) });
    outlier_objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 10.0, -1.0), radius: 5.0, material: bulb }));

    // Neat trick: embed a small sphere in another to simulate glass.  Might work by reversing coefficient also
    // let glass2 = Box::new(Dielectric { ref_index: 1.5 });
    // objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.0, -1.0), radius: 0.5, material: glass }));
    // objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.0, -1.0), radius: -0.45, material: glass2 }));

    // A gold wall
    let brushed_gold = Box::new(Metal { albedo: Vec3::new(1.0, 0.85, 0.0), fuzz: 0.3 });
    // let brushed_copper = Box::new(Metal { albedo: Vec3::new(0.7, 0.45, 0.2), fuzz: 0.3 });
    outlier_objects.push(Box::new(AARect { which: AARectWhich::XY, a_min: -4.0, a_max: 4.0, b_min: -4.0, b_max: 1.3, c: -1.7, negate_normal: false, material: brushed_gold}));

    // The giant world sphere on which all others sit
    let globe_texture = CheckerTexture {
        check_size: 10.0,
        odd: Box::new(ConstantTexture { colour: Vec3::new(0.2, 0.3, 0.1) }),
        even: Box::new(ConstantTexture { colour: Vec3::new(0.9, 0.9, 0.9) }),
    };
    outlier_objects.push(Box::new(Sphere {
        centre: world_centre,
        radius: world_radius,
        material: Box::new(TexturedLambertian { albedo: Box::new(globe_texture) }),
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
    let noise1 = Box::new(TexturedLambertian { albedo: Box::new(NoiseTexture::new(4.0, Vec3::new(1.0, 1.0, 1.0))) });
    outlier_objects.push(Box::new(Sphere {
        centre: Vec3::new(0.0, -1000.5, -2.0),
        radius: 1000.0,
        material: noise1,
    }));

    let marble = Box::new(PolishedStone { albedo: Box::new(NoiseTexture::new(12.0, Vec3::new(0.6, 0.2, 0.1))) });
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

fn main() {
    const COLS: usize = 400;
    const ROWS: usize = 400;
    const NUM_SAMPLES: usize = 100; // Sample code recommends 100 but this is slow
    const MAX_BOUNCES: usize = 30;
    const ASPECT_RATIO: f32 = COLS as f32 / ROWS as f32;

    println!("Welcome to JDs Rustaceous Raytracer!");

    // Set up variables for timing and progress printing
    let max_iterations = COLS * ROWS;
    let mut num_iterations = 0;
    let start_time = std::time::Instant::now();
    let mut last_time = std::time::Instant::now();
    let io_flush = || std::io::stdout().flush().expect("Could not flush stdout");
    io_flush();
    let mut total_rays = 0;

    // Generate the scene
    let scene_index = 1;
    let mut scene = match scene_index {
        0 => noise_scene(ASPECT_RATIO),
        _ => random_scene(ASPECT_RATIO),
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
                let ray = scene.camera.clip_to_ray(u, v);
                let (ray_colour, ray_count) = raytrace::cast_ray(&ray, &bvh, MAX_BOUNCES);
                colour += ray_colour;
                total_rays += ray_count;
            }
            colour *= 1.0 / NUM_SAMPLES as f32;
            // Clamp the colour to [0..1]
            let colour = colour.map(|x| clamp(x, 0.0, 1.0));
            // Gamma correction: sqrt the colour
            let colour = colour.map(|x| x.sqrt());
            // Output the colour for this pixel
            image.append_pixel(colour);

            num_iterations += 1;
            if last_time.elapsed().as_secs() >= 1 {
                let percentage_done = 100.0 * num_iterations as f32 / max_iterations as f32;
                let time_elapsed = post_scene_gen_time.elapsed();
                let time_elapsed = 1_000_000_000.0 * time_elapsed.as_secs() as f32 + time_elapsed.subsec_nanos() as f32;
                let time_per_ray = time_elapsed / total_rays as f32;
                print!("\rProcessed {:2.2}%;  {} bounces so far;  {:3.2}ns/ray                    ", percentage_done, total_rays, time_per_ray);
                io_flush();
                last_time += std::time::Duration::from_secs(1);
            }
        }
    }

    let time_elapsed = post_scene_gen_time.elapsed();
    let time_elapsed = 1_000_000_000.0 * time_elapsed.as_secs() as f32 + time_elapsed.subsec_nanos() as f32;
    let time_per_ray = time_elapsed / total_rays as f32;
    println!("\rProcessing done.                                                ");
    println!("    Rows:       {}", ROWS);
    println!("    Columns:    {}", COLS);
    println!("    Rays/pixel: {}", NUM_SAMPLES);
    println!("    ns/ray:     {:.1}", time_per_ray);
    println!("    Time taken: {:.3}", time_elapsed / 1_000_000_000.0);
    println!("    Rays cast:  {}", total_rays);
    println!("Writing file..");

    // Output the image to a file
    let path = Path::new("out/output.ppm");
    write_text_to_file(&image.get_text(), &path);
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
