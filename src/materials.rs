
use rand::{thread_rng,Rng};
use rand::distributions::Standard;
use noise::{NoiseFn,Perlin,Turbulence};

use super::math::*;

pub trait Texture {
    fn value(&self, u: f32, v: f32, p: Vec3) -> Vec3;
}

pub struct ConstantTexture {
    pub colour: Vec3,
}

impl Texture for ConstantTexture {
    fn value(&self, _u: f32, _v: f32, _p: Vec3) -> Vec3 {
        self.colour
    }
}

pub struct CheckerTexture {
    pub check_size: f32,
    pub odd: Box<Texture>,
    pub even: Box<Texture>,
}

impl Texture for CheckerTexture {
    fn value(&self, u: f32, v: f32, p: Vec3) -> Vec3 {
        let sines = (self.check_size * p).map(|x| x.sin());
        let sines = sines.x * sines.y * sines.z;
        if sines < 0.0 {
            self.odd.value(u, v, p)
        }
        else {
            self.even.value(u, v, p)
        }
    }
}

pub struct NoiseTexture {
    scale: f32,
    colour: Vec3,
    perlin: Perlin,
}

impl NoiseTexture {
    pub fn new(scale: f32, colour: Vec3) -> NoiseTexture {
        NoiseTexture { scale, colour, perlin: Perlin::new() }
    }
}

fn turb(noise: &Perlin, p: Vec3, depth: usize) -> f32 {
    let mut accum = 0.0;
    let mut temp_p = p;
    let mut weight = 1.0;
    for i in 0..depth {
        accum += weight * noise.get([temp_p.x as f64, temp_p.y as f64, temp_p.z as f64]);
        weight *= 0.5;
        temp_p *= 2.0;
    }
    accum.abs() as f32
}

impl Texture for NoiseTexture {
    fn value(&self, u: f32, v: f32, p: Vec3) -> Vec3 {

        // let noise = self.perlin.get([p.x as f64, p.y as f64, p.z as f64]);
        let noise = turb(&self.perlin, p, 7);
        // self.colour * 0.5 * (1.0 + noise as f32)
        self.colour * 0.5 * (1.0 + (self.scale * p.z + 10.0 * noise).sin())
    }
}
