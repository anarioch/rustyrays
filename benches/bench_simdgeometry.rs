
use raytrace::simdgeometry::*;

use std::hint::black_box;

use criterion::{Bencher, Criterion, criterion_group, criterion_main};

fn simdbench_ray_aabb_hit(b: &mut Bencher) {
    // Optionally include some setup
    let aabb = AABB { min: new_pos( -2.0, -2.0, -2.0), max: new_pos(2.0, -1.5, 2.0) };
    let origin = new_pos(0.0, 0.0, 0.0);
    let down_y = Ray { origin, direction: new_dir(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(aabb.hit(black_box(&down_y), 0.0, 1000.0));
    });
}

fn simdbench_ray_aabb_miss(b: &mut Bencher) {
    // Optionally include some setup
    let aabb = AABB { min: new_pos( -2.0, -2.0, -2.0), max: new_pos(2.0, -1.5, 2.0) };
    let origin = new_pos(3.0, 0.0, 3.0);
    let down_y = Ray { origin, direction: new_dir(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(aabb.hit(black_box(&down_y), 0.0, 1000.0));
    });
}

fn simdbench_ray_aabb_miss_by_t(b: &mut Bencher) {
    // Optionally include some setup
    let aabb = AABB { min: new_pos( -2.0, -2.0, -2.0), max: new_pos(2.0, -1.5, 2.0) };
    let origin = new_pos(0.0, 0.0, 0.0);
    let down_y = Ray { origin, direction: new_dir(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(aabb.hit(black_box(&down_y), 0.0, 0.9));
    });
}

fn aabb_scene() -> Vec<AABB> {
    let mut objects : Vec<AABB> = Vec::new();

    let radius = new_dir(0.2, 0.2, 0.2);
    for a in -7..7 {
        for b in -7..7 {
            let centre = new_pos(a as f32, 0.0, b as f32);
            let min = unsafe { _mm_sub_ps(centre, radius) };
            let max = unsafe { _mm_add_ps(centre, radius) };
            objects.push(AABB { min, max });
        }
    }

    objects
}

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn simdbench_ray_spherescene_aabbarray_hit(b: &mut Bencher) {
    // Given: a grid of objects
    let aabbs = aabb_scene();
    // let mut aabbs: Vec<AABB> = Vec::with_capacity(scene.len());
    // for obj in &scene {
    //     aabbs.push(obj.bounds().unwrap());
    // }

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: new_pos(0.5, 0.0, 0.0), direction: new_dir(1.0, 0.0, 0.0) };

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

fn grid_scene() -> Vec<Box<dyn raytrace::geometry::Hitable>> {
    let mut objects : Vec<Box<dyn raytrace::geometry::Hitable>> = Vec::new();

    for a in -7..7 {
        for b in -7..7 {
            let centre = raytrace::math::Vec3::new(a as f32, 0.0, b as f32);
            objects.push(Box::new(raytrace::geometry::Sphere { centre, radius: 0.2, material: raytrace::materials::Material::Invisible }));
        }
    }

    objects
}

fn simdbench_ray_spherescene_bvh_hit(b: &mut Bencher) {
    // Given: a grid of objects
    let mut scene = grid_scene();
    let bvh = raytrace::geometry::SIMDBVH::build(&mut scene);

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: new_pos(0.5, 0.0, 0.0), direction: new_dir(1.0, 0.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(bvh.hit(black_box(&ray_x_axis), 0.001, 1000.0));
    });
}

fn simdbench_ray_spherescene_bvh_hit2(b: &mut Bencher) {
    // Given: a grid of objects
    let mut scene = grid_scene();
    let bvh = raytrace::geometry::SIMDBVH::build(&mut scene);

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: new_pos(0.5, 0.0, 0.0), direction: new_dir(1.0, 0.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(bvh.hit(black_box(&ray_x_axis), 0.001, 1000.0));
    });
}

fn simdbench_ray_spherescene_bvh_miss(b: &mut Bencher) {
    // Given: a grid of objects
    let mut scene = grid_scene();
    let bvh = raytrace::geometry::SIMDBVH::build(&mut scene);

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: new_pos(0.5, 2.0, 0.0), direction: new_dir(1.0, 0.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        black_box(bvh.hit(black_box(&ray_x_axis), 0.001, 1000.0));
    });
}


fn benches(c: &mut Criterion) {
    c.bench_function("simd_ray_aabb_hit", simdbench_ray_aabb_hit);
    c.bench_function("simd_ray_aabb_miss", simdbench_ray_aabb_miss);
    c.bench_function("simd_ray_aabb_miss_by_t", simdbench_ray_aabb_miss_by_t);
    c.bench_function("simd_ray_spherescene_aabbarray_hit", simdbench_ray_spherescene_aabbarray_hit);
    c.bench_function("simd_ray_spherescene_bvh_hit", simdbench_ray_spherescene_bvh_hit);
    c.bench_function("simd_ray_spherescene_bvh_hit2", simdbench_ray_spherescene_bvh_hit2);
    c.bench_function("simd_ray_spherescene_bvh_miss", simdbench_ray_spherescene_bvh_miss);
}

criterion_group!(group, benches);
criterion_main!(group);
