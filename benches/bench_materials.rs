
use raytrace::math::*;
use raytrace::materials::*;
use raytrace::geometry::*;

use std::hint::black_box;

use criterion::{Bencher, Criterion, criterion_group, criterion_main};

fn bench_noise_texture(b: &mut Bencher) {
    // Optionally include some setup
    let texture = NoiseTexture::new(2.0, Vec3::new(1.0, 0.0, 0.0));

    b.iter(|| {
        // Inner closure, the actual test
        black_box(texture.value(0.0, 0.0, black_box(Vec3::new(0.0, 1.0, 1.0))));
    });
}

fn bench_lambertian(b: &mut Bencher) {
    // Optionally include some setup
    let material = Material::Lambertian { albedo: Vec3::new(1.0, 0.0, 0.0) };
    let ray = Ray { origin: Vec3::new(1.0, 1.0, 0.0), direction: Vec3::new(-1.0, 0.0, 0.0) };
    let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(scatter(black_box(&ray), black_box(&hit)).unwrap());
    });
}

fn bench_textured_lambertian(b: &mut Bencher) {
    // Optionally include some setup
    let texture = ConstantTexture { colour: Vec3::new(1.0, 0.0, 0.0) };
    let material = Material::TexturedLambertian { albedo: Box::new(texture) };
    let ray = Ray { origin: Vec3::new(1.0, 1.0, 0.0), direction: Vec3::new(-1.0, 0.0, 0.0) };
    let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(scatter(black_box(&ray), black_box(&hit)).unwrap());
    });
}

fn bench_metal(b: &mut Bencher) {
    // Optionally include some setup
    let material = Material::Metal { albedo: Vec3::new(1.0, 0.0, 0.0), fuzz: 0.0 };
    let ray = Ray { origin: Vec3::new(-1.0, 2.0, 0.0), direction: Vec3::new(1.0, -1.0, 0.0) };
    let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(scatter(black_box(&ray), black_box(&hit)).unwrap());
    });
}

fn bench_dielectric(b: &mut Bencher) {
    // Optionally include some setup
    let material = Material::Dielectric { ref_index: 1.5 };
    let ray = Ray { origin: Vec3::new(-1.0, 2.0, 0.0), direction: Vec3::new(1.0, -1.0, 0.0) };
    let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(scatter(black_box(&ray), black_box(&hit)).unwrap());
    });
}

fn benches(c: &mut Criterion) {
    c.bench_function("noise_texture", bench_noise_texture);
    c.bench_function("lambertian", bench_lambertian);
    c.bench_function("textured_lambertian", bench_textured_lambertian);
    c.bench_function("metal", bench_metal);
    c.bench_function("dielectric", bench_dielectric);
}

criterion_group!(group, benches);
criterion_main!(group);
