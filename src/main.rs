#![deny(bare_trait_objects)]

extern crate raytrace;
extern crate rand;
extern crate serde;
extern crate serde_json;

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;

use rand::Rng;

use serde::Deserialize;

use raytrace::math::*;
use raytrace::ppm::PpmImage;
use raytrace::geometry::*;
use raytrace::scene::*;

#[derive(Debug, Deserialize)]
struct RenderParameters {
    cols: usize,
    rows: usize,
    samples_per_pixel: usize,
    max_bounces: u32
}

fn read_params_from_file<P: AsRef<Path>>(path: P) -> Result<RenderParameters, Box<dyn Error>> {
    // Open the file in read-only mode with buffer.
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `RenderParameters`.
    let params = serde_json::from_reader(reader)?;

    // Return the `RenderParameters`.
    Ok(params)
}



fn clamp(mut x: f32, min: f32, max: f32) -> f32 {
    if x < min { x = min; }
    if x > max { x = max; }
    x
}

#[derive(Debug, Clone, Copy)]
struct PixelCell {
    colour: Vec3,
    num_rays: u32,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let params_path = &args[1];
    let scene_path = &args[2];

    println!("Welcome to JDs Rustaceous Raytracer!");

    // Load the scene
    let params: RenderParameters = read_params_from_file(params_path).unwrap();
    let aspect_ratio: f32 = params.cols as f32 / params.rows as f32;
    let mut scene = load_scene(aspect_ratio, scene_path);

    // Uncomment to output a file that writes out a randomised sphere grid
    // gen_sphere_grid();

    // Set up variables for timing and progress printing
    let max_iterations = params.cols * params.rows;
    let mut num_iterations: u64 = 0;
    let start_time = std::time::Instant::now();
    let mut last_time = std::time::Instant::now();
    let io_flush = || std::io::stdout().flush().expect("Could not flush stdout");
    io_flush();
    let mut total_rays: u64 = 0;

    // Split outliers from rest of objects here to improve AABB sizes
    // TODO Figure out better heuristics to this automatically
    let obj_slice = &mut scene.objects;
    let mut bvh = BVH::build(obj_slice);
    for ref outlier in &scene.outlier_objects {
        bvh = BVH::insert(bvh, outlier);
    }
    let bvh = bvh;

    println!("Scene generation done (Took {:.2}s)", start_time.elapsed().as_secs());
    let post_scene_gen_time = std::time::Instant::now();

    // Cast rays to generate the image
    let mut rng = rand::thread_rng();
    let mut accum : Vec<PixelCell> = Vec::with_capacity(params.cols * params.rows);
    accum.resize(params.cols * params.rows, PixelCell { colour: Vec3::splat(0.0), num_rays: 0 });
    for s in 0..params.samples_per_pixel {
        let mut img : Vec<PixelCell> = Vec::with_capacity(params.cols * params.rows);
        img.resize(params.cols * params.rows, PixelCell { colour: Vec3::splat(0.0), num_rays: 0 });
        for r in 0..params.rows {
            let pv = r as f32;
            for c in 0..params.cols {
                let pu = c as f32;
                // Anti-aliased: average colour from multiple randomised samples per pixel
                let u = (pu + rng.gen::<f32>()) / params.cols as f32;
                let v = (pv + rng.gen::<f32>()) / params.rows as f32;
                let ray = scene.camera.clip_to_ray(u, v);
                let (ray_colour, ray_count) = raytrace::cast_ray(&ray, &bvh, params.max_bounces);
                total_rays += ray_count as u64;

                // Store the colour
                img[params.cols * r + c] = PixelCell { colour: ray_colour, num_rays: ray_count };
            }

            num_iterations += params.cols as u64;

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

        // Accumulate this pass into the overall image
        for r in (0..params.rows).rev() {
            for c in 0..params.cols {
                let src = &img[params.cols * r + c];
                let target = &mut accum[params.cols * r + c];
                target.colour += src.colour;
                target.num_rays += src.num_rays;
            }
        }

        // Render the scene and write to a file
        {
            let mut image = PpmImage::create(params.cols, params.rows);
            // let mut ray_view = PpmImage::create(params.cols, params.rows);
            for r in (0..params.rows).rev() {
                for c in 0..params.cols {
                    // Add the colour to our accumulator
                    let cell = accum[params.cols * r + c];

                    // Output the colour to current image
                    let mut colour = cell.colour;
                    colour *= 1.0 / (s + 1) as f32;
                    // Clamp the colour to [0..1]
                    let colour = colour.map(|x| clamp(x, 0.0, 1.0));
                    // Gamma correction: sqrt the colour
                    let colour = colour.map(|x| x.sqrt());
                    image.append_pixel(colour);

                    // Output the ray-count to its image
                    // let ray_count_colour = cell.num_rays as f32 / (10.0 * (s + 1) as f32);
                    // ray_view.append_pixel(Vec3::splat(clamp(ray_count_colour, 0.0, 1.0)));
                }
            }

            let path = Path::new("out/output.ppm");
            write_text_to_file(&image.get_text(), &path, false);

            // let rays_path = Path::new("out/ray_counts.ppm");
            // write_text_to_file(&ray_view.get_text(), &rays_path, false);
        };
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
