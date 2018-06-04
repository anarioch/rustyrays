
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
