// Needed for using 'cargo bench', though I don't fully follow why
#![feature(test)]
extern crate test;

extern crate raytrace;

use raytrace::math::*;
use raytrace::materials::*;
use raytrace::geometry::*;

use test::{Bencher,black_box};

#[bench]
fn bench_noise_texture(b: &mut Bencher) {
    // Optionally include some setup
    let texture = NoiseTexture::new(2.0, Vec3::new(1.0, 0.0, 0.0));

    b.iter(|| {
        // Inner closure, the actual test
        texture.value(0.0, 0.0, black_box(Vec3::new(0.0, 1.0, 1.0)));
    });
}

#[bench]
fn bench_lambertian(b: &mut Bencher) {
    // Optionally include some setup
    let material = Lambertian { albedo: Vec3::new(1.0, 0.0, 0.0) };
    let ray = Ray { origin: Vec3::new(1.0, 1.0, 0.0), direction: Vec3::new(-1.0, 0.0, 0.0) };
    let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

    b.iter(|| {
        // Inner closure, the actual test
        material.scatter(black_box(&ray), black_box(&hit)).unwrap();
    });
}

#[bench]
fn bench_textured_lambertian(b: &mut Bencher) {
    // Optionally include some setup
    let texture = ConstantTexture { colour: Vec3::new(1.0, 0.0, 0.0) };
    let material = TexturedLambertian { albedo: Box::new(texture) };
    let ray = Ray { origin: Vec3::new(1.0, 1.0, 0.0), direction: Vec3::new(-1.0, 0.0, 0.0) };
    let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

    b.iter(|| {
        // Inner closure, the actual test
        material.scatter(black_box(&ray), black_box(&hit)).unwrap();
    });
}

#[bench]
fn bench_metal(b: &mut Bencher) {
    // Optionally include some setup
    let material = Metal { albedo: Vec3::new(1.0, 0.0, 0.0), fuzz: 0.0 };
    let ray = Ray { origin: Vec3::new(-1.0, 2.0, 0.0), direction: Vec3::new(1.0, -1.0, 0.0) };
    let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

    b.iter(|| {
        // Inner closure, the actual test
        material.scatter(black_box(&ray), black_box(&hit)).unwrap();
    });
}

#[bench]
fn bench_dielectric(b: &mut Bencher) {
    // Optionally include some setup
    let material = Dielectric { ref_index: 1.5 };
    let ray = Ray { origin: Vec3::new(-1.0, 2.0, 0.0), direction: Vec3::new(1.0, -1.0, 0.0) };
    let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

    b.iter(|| {
        // Inner closure, the actual test
        material.scatter(black_box(&ray), black_box(&hit)).unwrap();
    });
}
