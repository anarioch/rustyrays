
use rand::{thread_rng,Rng};
use rand::distributions::Standard;

use super::math::Vec3;

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

pub struct Perlin {
    random_data: Vec<f32>,
    perm_x: Vec<usize>,
    perm_y: Vec<usize>,
    perm_z: Vec<usize>,
}

impl Perlin {
    fn generate_randoms() -> Vec<f32> {
        thread_rng().sample_iter(&Standard).take(256).collect()
    }
    fn generate_permutation() -> Vec<usize> {
        let mut indices = (0..256).collect::<Vec<usize>>();
        thread_rng().shuffle(&mut indices);
        indices
    }
    fn trilinear_interp(c: [[[f32; 2]; 2]; 2], coeff: Vec3) -> f32 {
        let mut accum = 0.0;
        let lerp = |i,x| i as f32 * x + (1 - i) as f32 * (1.0 - x);
        for i in 0..2 {
            let mi = lerp(i, coeff.x);
            for j in 0..2 {
                let mj = lerp(j, coeff.y);
                for k in 0..2 {
                    let mk = lerp(k, coeff.z);
                    accum += mi * mj * mk * c[i][j][k];
                }
            }
        }
        accum
    }

    pub fn new() -> Perlin {
        let random_data = Perlin::generate_randoms();
        let perm_x = Perlin::generate_permutation();
        let perm_y = Perlin::generate_permutation();
        let perm_z = Perlin::generate_permutation();
        Perlin { random_data, perm_x, perm_y, perm_z }
    }
    fn noise(&self, p: Vec3) -> f32 {
        let base = p.map(|x| x.floor());
        let coeff = p - base;
        let coeff = coeff.map(|x| x * x * (3.0 - 2.0 * x));
        let mask = |x,i| ((x as i32+i as i32) & 255) as usize;
        let mut c = [[[0.0; 2]; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let index = self.perm_x[mask(base.x, i)] ^ self.perm_y[mask(base.y, j)] ^ self.perm_z[mask(base.z, k)];
                    c[i][j][k] = self.random_data[index];
                }
            }
        }
        Perlin::trilinear_interp(c, coeff)
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

impl Texture for NoiseTexture {
    fn value(&self, u: f32, v: f32, p: Vec3) -> Vec3 {
        self.colour * self.perlin.noise(self.scale * p)
    }
}
