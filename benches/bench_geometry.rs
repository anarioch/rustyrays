// Needed for using 'cargo bench', though I don't fully follow why
#![feature(test)]
extern crate test;

extern crate raytrace;

use raytrace::math::*;
use raytrace::materials::Invisible;
use raytrace::geometry::*;

use test::Bencher;

#[bench]
fn bench_ray_aabb_hit(b: &mut Bencher) {
    // Optionally include some setup
    let aabb = AABB { min: Vec3::new( -2.0, -2.0, -2.0), max: Vec3::new(2.0, -1.5, 2.0) };
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let down_y = Ray { origin, direction: Vec3::new(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        aabb.hit(&down_y, 0.0, 1000.0);
    });
}

#[bench]
fn bench_ray_aabb_miss(b: &mut Bencher) {
    // Optionally include some setup
    let aabb = AABB { min: Vec3::new( -2.0, -2.0, -2.0), max: Vec3::new(2.0, -1.5, 2.0) };
    let origin = Vec3::new(3.0, 0.0, 3.0);
    let down_y = Ray { origin, direction: Vec3::new(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        aabb.hit(&down_y, 0.0, 1000.0);
    });
}

#[bench]
fn bench_ray_aabb_miss_by_t(b: &mut Bencher) {
    // Optionally include some setup
    let aabb = AABB { min: Vec3::new( -2.0, -2.0, -2.0), max: Vec3::new(2.0, -1.5, 2.0) };
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let down_y = Ray { origin, direction: Vec3::new(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        aabb.hit(&down_y, 0.0, 0.9);
    });
}

#[bench]
fn bench_ray_sphere_hit(b: &mut Bencher) {
    // Optionally include some setup
    let sphere = Sphere { centre: Vec3::new(0.0, -2.0, 0.0), radius: 1.0, material: Box::new(Invisible {}) };
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let down_y = Ray { origin, direction: Vec3::new(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        sphere.hit(&down_y, 0.0, 1000.0).unwrap();
    });
}

#[bench]
fn bench_ray_sphere_miss(b: &mut Bencher) {
    // Optionally include some setup
    let sphere = Sphere { centre: Vec3::new(0.0, -2.0, 0.0), radius: 1.0, material: Box::new(Invisible {}) };
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let parallel_y = Ray { origin, direction: Vec3::new(2.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        sphere.hit(&parallel_y, 0.0, 1000.0).is_none();
    });
}

#[bench]
fn bench_ray_sphere_miss_by_t(b: &mut Bencher) {
    // Optionally include some setup
    let sphere = Sphere { centre: Vec3::new(0.0, -2.0, 0.0), radius: 1.0, material: Box::new(Invisible {}) };
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let down_y = Ray { origin, direction: Vec3::new(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        sphere.hit(&down_y, 0.0, 0.9).is_none();
    });
}

#[bench]
fn bench_hit_aarect(b: &mut Bencher) {
    // Optionally include some setup
    let rect = AARect { which: AARectWhich::XZ, a_min: -2.0, a_max: 2.0, b_min: -2.0, b_max: 2.0, c: -2.0, negate_normal: true, material: Box::new(Invisible {}) };
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let down_y = Ray { origin, direction: Vec3::new(0.0, -1.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        rect.hit(&down_y, 0.0, 1000.0).unwrap();
    });
}


fn grid_scene() -> Vec<Box<Hitable>> {
    let mut objects : Vec<Box<Hitable>> = Vec::new();

    for a in -7..7 {
        for b in -7..7 {
            let centre = Vec3::new(a as f32, 0.0, b as f32);
            objects.push(Box::new(Sphere { centre, radius: 0.2, material: Box::new(Invisible {}) }));
        }
    }

    objects
}

#[bench]
fn bench_ray_spheres_intersect(b: &mut Bencher) {
    // Given: a grid of objects
    let scene = grid_scene();

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: Vec3::new(0.5, 0.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        hit(&ray_x_axis, 0.001, 1000.0, &scene);
    });
}

#[bench]
fn bench_ray_spheres_intersect_bvh(b: &mut Bencher) {
    // Given: a grid of objects
    let mut scene = grid_scene();
    let bvh = BVH::build(&mut scene);

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: Vec3::new(0.5, 0.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };

    b.iter(|| {
        // Inner closure, the actual test
        bvh.hit(&ray_x_axis, 0.001, 1000.0);
    });
}
