use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;

use camera::*;
use geometry::*;
use materials::*;
use math::*;

use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct CameraDeclaration {
    eye: Vec3,
    focus: Vec3,
    up: Vec3,
    vertical_fov: f32,
    aperture: f32
}

#[derive(Debug, Deserialize, Serialize)]
enum AARectWhichDecl {
    XY,
    XZ,
    YZ,
}

#[derive(Debug, Deserialize, Serialize)]
struct SphereDeclaration {
    centre: Vec3,
    radius: f32,
    material: String
}

#[derive(Debug, Deserialize, Serialize)]
struct AARectDeclaration {
    which: AARectWhichDecl,
    a_min: f32,
    a_max: f32,
    b_min: f32,
    b_max: f32,
    c: f32,
    negate_normal: bool,
    material: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase", tag = "shape")]
enum ShapeDeclaration {
    Sphere(SphereDeclaration),
    AARect(AARectDeclaration)
}

#[derive(Debug, Deserialize, Serialize)]
struct SceneDeclaration {
    objects: Vec<ShapeDeclaration>,
    outlier_objects: Vec<ShapeDeclaration>,
    camera: CameraDeclaration
}


/// Definition of a scene to be rendered, including objects and camera.
/// Objects are split into two groups, to provide hints for BVH construction
pub struct Scene {
    pub objects: Vec<Box<dyn Hitable>>,
    pub outlier_objects: Vec<Box<dyn Hitable>>,
    pub camera: Camera,
}

fn read_spec_from_file<P: AsRef<Path>>(path: P) -> Result<SceneDeclaration, Box<dyn Error>> {
    // Open the file in read-only mode with buffer.
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `SceneDeclaration`.
    let scene = serde_json::from_reader(reader)?;

    // Return the `SceneDeclaration`.
    Ok(scene)
}

pub fn gen_sphere_grid() {
    let mut rng = rand::thread_rng();
    let mut rand = || rng.gen::<f32>();

    let world_radius = 1000.0;

    // Create closure that creates a randomised sphere within the x,z unit cell
    let rad_sq = world_radius * world_radius;
    let mut random_sphere = |x, z| {
        // let radius = 0.2;
        let radius = 0.15 + 0.1 * rand();
        let mut centre = Vec3::new(x + 0.9 * rand(), 0.0, z + 0.9 * rand());
        centre.y = (rad_sq - centre.x * centre.x).sqrt() - world_radius + radius;
        let mat = match rand() {
            d if d < 0.65 => "rand_lambertian",
            d if d < 0.85 => "rand_metal",
            _ => "glass",
        };
        SphereDeclaration {centre, radius, material: mat.to_owned()}
    };

    let mut dec: Vec<SphereDeclaration> = Vec::new();
    for a in -7..7 {
        for b in -7..7 {
            dec.push(random_sphere(a as f32, b as f32));
        }
    }

    let ss = serde_json::to_string(&dec).unwrap();
    let path = Path::new("sphere_grid.json");
    write_text_to_file(&ss, &path);
}

pub fn load_scene<P: AsRef<Path>>(aspect_ratio: f32, scene_path: P) -> Scene {
    let scene_spec: SceneDeclaration = read_spec_from_file(scene_path).unwrap();

    let mut objects : Vec<Box<dyn Hitable>> = Vec::new();
    let mut outlier_objects : Vec<Box<dyn Hitable>> = Vec::new();
    let mut rng = rand::thread_rng();
    let mut rand = || rng.gen::<f32>();

    let mut m = |mat: &str| match mat {
        "gold" => Material::Metal { albedo: Vec3::new(0.8, 0.6, 0.2), fuzz: 0.0 },
        "brushed_gold" => Material::Metal { albedo: Vec3::new(1.0, 0.85, 0.0), fuzz: 0.3 },
        "marble" => Material::PolishedStone { albedo: Box::new(NoiseTexture::new(12.0, Vec3::new(0.6, 0.1, 0.2))) },
        "glass" => Material::Dielectric { ref_index: 1.5 },
        "green_checker" => Material::TexturedLambertian { albedo: Box::new(CheckerTexture { check_size: 10.0, odd: Vec3::new(0.2, 0.3, 0.1), even: Vec3::splat(0.9) }) },
        "rand_lambertian" => Material::Lambertian { albedo: Vec3::new(rand() * rand(), rand() * rand(), rand() * rand()) },
        "white" => Material::Lambertian { albedo: Vec3::splat(1.0) },
        "black" => Material::Lambertian { albedo: Vec3::splat(0.0) },
        "red" => Material::Lambertian { albedo: Vec3::new(1.0, 0.0, 0.0) },
        "green" => Material::Lambertian { albedo: Vec3::new(0.0, 1.0, 0.0) },
        "blue" => Material::Lambertian { albedo: Vec3::new(0.0, 0.0, 1.0) },
        "rand_metal" => Material::Metal { albedo: Vec3::new(0.5 * (1.0 + rand()), 0.5 * (1.0 + rand()), 0.5 * (1.0 + rand())), fuzz: 0.0 },
        "rand_brushed_metal" => Material::Metal { albedo: Vec3::new(0.5 * (1.0 + rand()), 0.5 * (1.0 + rand()), 0.5 * (1.0 + rand())), fuzz: 0.5 * rand() },
        "glow_white" => Material::DiffuseLight { emission_colour: Vec3::splat(4.0) },
        _ => Material::Lambertian { albedo: Vec3::splat(1.0) }
    };
    for obj in &scene_spec.objects {
        match obj {
            ShapeDeclaration::Sphere(sphere) => 
                objects.push(Box::new(Sphere { centre: sphere.centre, radius: sphere.radius, material: m(sphere.material.as_ref()) })),
            ShapeDeclaration::AARect(rect) =>
                outlier_objects.push(Box::new(AARect { which: match rect.which { AARectWhichDecl::XY => AARectWhich::XY, AARectWhichDecl::XZ => AARectWhich::XZ, AARectWhichDecl::YZ => AARectWhich::YZ }, a_min: rect.a_min, a_max: rect.a_max, b_min: rect.b_min, b_max: rect.b_max, c: rect.c, negate_normal: rect.negate_normal, material: m(rect.material.as_ref())}))
        }
    }

    // Neat trick: embed a small sphere in another to simulate glass.  Might work by reversing coefficient also
    // let glass2 = Material::Dielectric { ref_index: 1.5 };
    // objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.0, -1.0), radius: 0.5, material: glass }));
    // objects.push(Box::new(Sphere { centre: Vec3::new(0.0, 0.0, -1.0), radius: -0.45, material: glass2 }));

    for obj in &scene_spec.outlier_objects {
        match obj {
            ShapeDeclaration::Sphere(sphere) => 
                objects.push(Box::new(Sphere { centre: sphere.centre, radius: sphere.radius, material: m(sphere.material.as_ref()) })),
            ShapeDeclaration::AARect(rect) =>
                outlier_objects.push(Box::new(AARect { which: match rect.which { AARectWhichDecl::XY => AARectWhich::XY, AARectWhichDecl::XZ => AARectWhich::XZ, AARectWhichDecl::YZ => AARectWhich::YZ }, a_min: rect.a_min, a_max: rect.a_max, b_min: rect.b_min, b_max: rect.b_max, c: rect.c, negate_normal: rect.negate_normal, material: m(rect.material.as_ref())}))
        }
    }

    // Configure the camera
    let camera_spec = &scene_spec.camera;
    let dist_to_focus = (camera_spec.focus - camera_spec.eye).length();
    let camera = Camera::new(camera_spec.eye, camera_spec.focus, camera_spec.up, camera_spec.vertical_fov, aspect_ratio, camera_spec.aperture, dist_to_focus);

    Scene { objects, outlier_objects, camera }
}

fn write_text_to_file(text: &str, path: &Path) {
    let display = path.display();

    let mut file = match File::create(&path) {
        Err(why) => panic!("Failed to create file {}: {}", display, why),
        Ok(file) => file,
    };

    match file.write_all(text.as_bytes()) {
        Err(why) => panic!("Failed to write to file {}: {}", display, why),
        Ok(_) => println!("Wrote to file {}!", display),
    }
}
