// Needed for using 'cargo bench', though I don't fully follow why
#![feature(test)]
extern crate test;

extern crate float_cmp;

extern crate raytrace;

use float_cmp::ApproxEqUlps;

use raytrace::math::*;
use raytrace::materials::*;
use raytrace::geometry::*;

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

#[test]
fn ray_spheres_intersect() {
    // Given: a grid of objects
    let scene = grid_scene();

    // Given: a few rays that do or don't intersect a/some/many objects
    let ray_x_axis = Ray { origin: Vec3::new(0.5, 0.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };
    let ray_above_scene = Ray { origin: Vec3::new(0.5, 2.0, 0.0), direction: Vec3::new(1.0, 0.0, 0.0) };
    let ray_y_axis = Ray { origin: Vec3::new(0.0, 2.0, 0.0), direction: Vec3::new(0.0, -1.0, 0.0) };
    let ray_diag_hit = Ray { origin: Vec3::new(-1.0, -1.0, 0.0), direction: Vec3::new(1.0, 1.0, 0.0) };
    let ray_diag_miss = Ray { origin: Vec3::new(-1.5, -1.0, -1.5), direction: Vec3::new(0.2, 1.0, 0.0) };

    // When: we cast the X Axis ray
    let res = hit(&ray_x_axis, 0.001, 1000.0, &scene);
    // Then: it hit something
    let res = res.unwrap();
    assert!(res.t.approx_eq_ulps(&0.3, 2));

    // When: we cast the ray above the scene
    let res = hit(&ray_above_scene, 0.001, 1000.0, &scene);
    // Then: it hits nothing
    assert!(res.is_none());

    // When: we cast the ray down the Y axis
    let res = hit(&ray_y_axis, 0.001, 1000.0, &scene);
    // Then: it hits the sphere in the centre
    let res = res.unwrap();
    assert!(res.t.approx_eq_ulps(&1.8, 2));

    // When: we cast the first diagonal ray
    let res = hit(&ray_diag_hit, 0.001, 1000.0, &scene);
    // Then: it hits the sphere in the centre
    let res = res.unwrap();
    assert!(res.t.approx_eq_ulps(&0.8585787, 2));

    // When: we cast the second diagonal ray
    let res = hit(&ray_diag_miss, 0.001, 1000.0, &scene);
    // Then: it hits the sphere in the centre
    assert!(res.is_none());
}
