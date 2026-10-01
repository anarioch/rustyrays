
use raytrace::math::*;
use raytrace::geometry::*;
use raytrace::materials::Material;

use std::hint::black_box;

use criterion::{Bencher, Criterion, criterion_group, criterion_main};

fn bench_ray_aabb_hit(b: &mut Bencher) {
    // Optionally include some setup
    let aabb = AABB { min: Vec3::new( -2.0, -2.0, -2.0), max: Vec3::new(2.0, -1.5, 2.0) };
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let down_y = Ray { origin, direction: Vec3::new(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(aabb.hit(black_box(&down_y), 0.0, 1000.0));
    });
}

fn bench_ray_aabb_miss(b: &mut Bencher) {
    // Optionally include some setup
    let aabb = AABB { min: Vec3::new( -2.0, -2.0, -2.0), max: Vec3::new(2.0, -1.5, 2.0) };
    let origin = Vec3::new(3.0, 0.0, 3.0);
    let down_y = Ray { origin, direction: Vec3::new(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(aabb.hit(black_box(&down_y), 0.0, 1000.0));
    });
}

fn bench_ray_aabb_miss_by_t(b: &mut Bencher) {
    // Optionally include some setup
    let aabb = AABB { min: Vec3::new( -2.0, -2.0, -2.0), max: Vec3::new(2.0, -1.5, 2.0) };
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let down_y = Ray { origin, direction: Vec3::new(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(aabb.hit(black_box(&down_y), 0.0, 0.9));
    });
}

fn bench_ray_sphere_hit(b: &mut Bencher) {
    // Optionally include some setup
    let sphere = Sphere { centre: Vec3::new(0.0, -2.0, 0.0), radius: 1.0, material: Material::Invisible };
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let down_y = Ray { origin, direction: Vec3::new(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(sphere.hit(black_box(&down_y), 0.0, 1000.0).unwrap());
    });
}

fn bench_ray_sphere_miss(b: &mut Bencher) {
    // Optionally include some setup
    let sphere = Sphere { centre: Vec3::new(0.0, -2.0, 0.0), radius: 1.0, material: Material::Invisible };
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let parallel_y = Ray { origin, direction: Vec3::new(2.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(sphere.hit(black_box(&parallel_y), 0.0, 1000.0).is_none());
    });
}

fn bench_ray_sphere_miss_by_t(b: &mut Bencher) {
    // Optionally include some setup
    let sphere = Sphere { centre: Vec3::new(0.0, -2.0, 0.0), radius: 1.0, material: Material::Invisible };
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let down_y = Ray { origin, direction: Vec3::new(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(sphere.hit(black_box(&down_y), 0.0, 0.9).is_none());
    });
}

fn bench_ray_aarect_hit(b: &mut Bencher) {
    // Optionally include some setup
    let rect = AARect { which: AARectWhich::XZ, a_min: -2.0, a_max: 2.0, b_min: -2.0, b_max: 2.0, c: -2.0, negate_normal: true, material: Material::Invisible };
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let down_y = Ray { origin, direction: Vec3::new(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(rect.hit(black_box(&down_y), 0.0, 1000.0).unwrap());
    });
}


fn grid_scene() -> Vec<Box<dyn Hitable>> {
    let mut objects : Vec<Box<dyn Hitable>> = Vec::new();

    for a in -7..7 {
        for b in -7..7 {
            let centre = Vec3::new(a as f32, 0.0, b as f32);
            objects.push(Box::new(Sphere { centre, radius: 0.2, material: Material::Invisible }));
        }
    }

    objects
}

fn bench_ray_spherescene_hit(b: &mut Bencher) {
    // Given: a grid of objects
    let scene = grid_scene();

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: Vec3::new(0.5, 0.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(hit(black_box(&ray_x_axis), 0.001, 1000.0, &scene));
    });
}

fn bench_ray_spherescene_bvh_hit(b: &mut Bencher) {
    // Given: a grid of objects
    let mut scene = grid_scene();
    let bvh = BVH::build(&mut scene);

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: Vec3::new(0.5, 0.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(bvh.hit(black_box(&ray_x_axis), 0.001, 1000.0));
    });
}

fn bench_ray_spherescene_bvh_hit2(b: &mut Bencher) {
    // Given: a grid of objects
    let mut scene = grid_scene();
    let bvh = BVH::build(&mut scene);

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: Vec3::new(0.5, 0.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(bvh.hit(black_box(&ray_x_axis), 0.001, 1000.0));
    });
}

fn bench_ray_spherescene_miss(b: &mut Bencher) {
    // Given: a grid of objects
    let scene = grid_scene();

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: Vec3::new(0.5, 2.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(hit(black_box(&ray_x_axis), 0.001, 1000.0, &scene));
    });
}

fn bench_ray_spherescene_bvh_miss(b: &mut Bencher) {
    // Given: a grid of objects
    let mut scene = grid_scene();
    let bvh = BVH::build(&mut scene);

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: Vec3::new(0.5, 2.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(bvh.hit(black_box(&ray_x_axis), 0.001, 1000.0));
    });
}

fn bench_ray_spherescene_naive_hit(b: &mut Bencher) {
    // Given: a grid of objects
    let scene = grid_scene();

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: Vec3::new(0.5, 0.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        let ray = black_box(&ray_x_axis);
        for obj in &scene {
            black_box(obj.hit(ray, 0.001, 1000.0));
        }
    });
}

fn bench_ray_spherescene_aabbarray_hit(b: &mut Bencher) {
    // Given: a grid of objects
    let scene = grid_scene();
    let mut aabbs: Vec<AABB> = Vec::with_capacity(scene.len());
    for obj in &scene {
        aabbs.push(obj.bounds().unwrap());
    }

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: Vec3::new(0.5, 0.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        let ray = black_box(&ray_x_axis);
        for obj in &aabbs {
            black_box(obj.hit(ray, 0.001, 1000.0));
        }
        // let candidates = aabbs.iter()
        //      .enumerate()
        //      .filter_map(|(i,aabb)| match aabb.hit(ray, 0.001, 1000.0) { true => Some(i), false => None });

        // candidates.collect::<Vec<usize>>();

        // let mut result = None;
        // let mut closest_so_far = 1000.0;
        // for index in candidates {
        //     let obj = &scene[index];
        //     if let Some(record) = (*obj).hit(&ray, 0.001, closest_so_far) {
        //         closest_so_far = record.t;
        //         result = Some(record);
        //     }
        // }

        // result;

    });
}


fn benches(c: &mut Criterion) {
    c.bench_function("ray_aabb_hit", bench_ray_aabb_hit);
    c.bench_function("ray_aabb_miss", bench_ray_aabb_miss);
    c.bench_function("ray_aabb_miss_by_t", bench_ray_aabb_miss_by_t);
    c.bench_function("ray_sphere_hit", bench_ray_sphere_hit);
    c.bench_function("ray_sphere_miss", bench_ray_sphere_miss);
    c.bench_function("ray_sphere_miss_by_t", bench_ray_sphere_miss_by_t);
    c.bench_function("ray_aarect_hit", bench_ray_aarect_hit);
    c.bench_function("ray_spherescene_hit", bench_ray_spherescene_hit);
    c.bench_function("ray_spherescene_bvh_hit", bench_ray_spherescene_bvh_hit);
    c.bench_function("ray_spherescene_bvh_hit2", bench_ray_spherescene_bvh_hit2);
    c.bench_function("ray_spherescene_miss", bench_ray_spherescene_miss);
    c.bench_function("ray_spherescene_bvh_miss", bench_ray_spherescene_bvh_miss);
    c.bench_function("ray_spherescene_naive_hit", bench_ray_spherescene_naive_hit);
    c.bench_function("ray_spherescene_aabbarray_hit", bench_ray_spherescene_aabbarray_hit);
}

criterion_group!(group, benches);
criterion_main!(group);
