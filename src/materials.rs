
use noise::{NoiseFn,Perlin};

use rand;
use rand::Rng;

use super::math::*;
use super::geometry::HitRecord;

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
    for _i in 0..depth {
        accum += weight * noise.get([f64::from(temp_p.x), f64::from(temp_p.y), f64::from(temp_p.z)]);
        weight *= 0.5;
        temp_p *= 2.0;
    }
    accum.abs() as f32
}

impl Texture for NoiseTexture {
    fn value(&self, _u: f32, _v: f32, p: Vec3) -> Vec3 {
        const NUM_FREQUENCIES: usize = 7;

        // let noise = self.perlin.get([p.x as f64, p.y as f64, p.z as f64]);
        let noise = turb(&self.perlin, p, NUM_FREQUENCIES);
        // self.colour * 0.5 * (1.0 + noise as f32)
        self.colour * 0.5 * (1.0 + (self.scale * p.z + 10.0 * noise).sin())
    }
}

pub struct ScatterResult {
    pub attenuation: Vec3,
    pub scattered: Ray,
}

pub trait Material {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<ScatterResult>;
    fn emit(&self) -> Vec3 {
        Vec3::new(0.0, 0.0, 0.0)
    }
}

pub struct Invisible {
    // Nothing here
}

impl Material for Invisible {
    fn scatter(&self, _ray: &Ray, _hit: &HitRecord) -> Option<ScatterResult>{
        None
    }
}

pub struct Lambertian {
    pub albedo: Vec3,
}

impl Material for Lambertian {
    fn scatter(&self, _ray: &Ray, hit: &HitRecord) -> Option<ScatterResult> {
        let target = hit.p + hit.normal + Vec3::random_in_unit_sphere(&mut rand::thread_rng());
        let dir = target - hit.p;
        let attenuation = self.albedo;
        let scattered = Ray { origin: hit.p, direction: dir };
        Some(ScatterResult { attenuation, scattered})
    }
}

pub struct TexturedLambertian {
    pub albedo: Box<Texture>,
}

impl Material for TexturedLambertian {
    fn scatter(&self, _ray: &Ray, hit: &HitRecord) -> Option<ScatterResult> {
        let target = hit.p + hit.normal + Vec3::random_in_unit_sphere(&mut rand::thread_rng());
        let dir = target - hit.p;
        let attenuation = self.albedo.value(0.0, 0.0, hit.p);
        let scattered = Ray { origin: hit.p, direction: dir };
        Some(ScatterResult { attenuation, scattered})
    }
}

pub struct Metal {
    pub albedo: Vec3,
    pub fuzz: f32,
}

impl Material for Metal {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<ScatterResult> {
        let reflected = reflect(ray.direction.normalise(), hit.normal);
        let reflected = reflected + self.fuzz * Vec3::random_in_unit_sphere(&mut rand::thread_rng());
        if dot(reflected, hit.normal) > 0.0 {
            let scattered = Ray { origin: hit.p, direction: reflected };
            Some(ScatterResult { attenuation: self.albedo, scattered})
        }
        else {
            None
        }
    }
}

pub struct PolishedStone {
    pub albedo: Box<Texture>,
}

impl Material for PolishedStone {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<ScatterResult> {
        let mut rng = rand::thread_rng();
        let reflected = reflect(ray.direction.normalise(), hit.normal);
        let dotty = dot(reflected, hit.normal);
        let reflect_prob = 1.0 - dotty.sqrt();
        let (attenuation, direction) = if rng.gen::<f32>() < reflect_prob {
            (Vec3::new(1.0, 1.0, 1.0), reflected)
        }
        else {
            let attenuation = self.albedo.value(0.0, 0.0, hit.p);
            let scattered = hit.normal + Vec3::random_in_unit_sphere(&mut rng);
            (attenuation, scattered)
        };
        let scattered = Ray { origin: hit.p, direction };
        Some(ScatterResult { attenuation, scattered})
    }
}

pub struct Dielectric {
    pub ref_index: f32,
}

fn schlick(cosine: f32, ref_index: f32) -> f32 {
    let r0 = (1.0 - ref_index) / (1.0 + ref_index);
    let r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

impl Material for Dielectric {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<ScatterResult> {
        let dir = ray.direction.normalise();
        let reflected = reflect(dir, hit.normal);
        let attenuation = Vec3::new(1.0, 1.0, 1.0);

        let ray_dot_norm = dot(ray.direction, hit.normal);
        let cosine = ray_dot_norm / ray.direction.len_sq().sqrt();
        let (outward_normal, ni_over_nt, cosine) =
            if ray_dot_norm > 0.0 {
                (-hit.normal, self.ref_index, self.ref_index * cosine)
            }
            else {
                (hit.normal, 1.0 / self.ref_index, -cosine)
            };
        
        let (refracted, reflect_prob) =
            match refract(dir, outward_normal, ni_over_nt) {
                Some(refracted) => {
                    (refracted, schlick(cosine, self.ref_index))
                },
                None => (Vec3::new(0.0, 0.0, 0.0), 1.0)
            };
        let ray_dir = 
            if rand::thread_rng().gen::<f32>() < reflect_prob {
                reflected
            }
            else {
                refracted
            };
        let scattered = Ray { origin: hit.p, direction: ray_dir };
        Some(ScatterResult { attenuation, scattered })
    }
}

pub struct DiffuseLight {
    pub emission_colour: Vec3,
}

impl Material for DiffuseLight {
    fn scatter(&self, _ray: &Ray, _hit: &HitRecord) -> Option<ScatterResult> {
        None
    }
    fn emit(&self) -> Vec3 {
        self.emission_colour
    }
}

#[cfg(test)]
mod tests {
    use super::super::math::*;
    use super::super::geometry::HitRecord;
    use super::*;

    use test::Bencher;

    #[test]
    fn scatter_lambertian() {
        // Given:
        let material = Lambertian { albedo: Vec3::new(1.0, 0.0, 0.0) };
        let ray = Ray { origin: Vec3::new(1.0, 1.0, 0.0), direction: Vec3::new(-1.0, 0.0, 0.0) };
        let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

        // When: We scatter off the material
        let res = material.scatter(&ray, &hit);

        // Then: the result is not None
        let res = res.unwrap();

        // Then: the attenuation is the red that we defined on the material
        assert_eq!(res.attenuation, material.albedo);
    }

    #[test]
    fn scatter_metal() {
        // Given:
        let material = Metal { albedo: Vec3::new(1.0, 0.0, 0.0), fuzz: 0.0 };
        let ray = Ray { origin: Vec3::new(-1.0, 2.0, 0.0), direction: Vec3::new(1.0, -1.0, 0.0) };
        let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

        // When: We scatter off the material
        let res = material.scatter(&ray, &hit);

        // Then: the result is not None
        let res = res.unwrap();

        // Then: the attenuation is the red that we defined on the material
        assert_eq!(res.attenuation, material.albedo);
        assert_eq!(res.scattered.origin, hit.p);
        assert_eq!(res.scattered.direction, Vec3::new(1.0, 1.0, 0.0) * (1.0 / f32::sqrt(2.0)));
    }

    #[test]
    fn scatter_dielectric() {
        // Given:
        let material = Dielectric { ref_index: 1.5 };
        let ray = Ray { origin: Vec3::new(-1.0, 2.0, 0.0), direction: Vec3::new(1.0, -1.0, 0.0) };
        let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

        // When: We scatter off the material
        let res = material.scatter(&ray, &hit);

        // Then: the result is not None
        let res = res.unwrap();

        // Then: the attenuation is the red that we defined on the material
        assert_eq!(res.attenuation, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(res.scattered.origin, hit.p);
        assert_eq!(res.scattered.direction, Vec3::new(0.47140455, -0.8819171, 0.0));
    }

    #[bench]
    fn bench_noise_texture(b: &mut Bencher) {
        // Optionally include some setup
        let texture = NoiseTexture::new(2.0, Vec3::new(1.0, 0.0, 0.0));

        b.iter(|| {
            // Inner closure, the actual test
            texture.value(0.0, 0.0, Vec3::new(0.0, 1.0, 1.0));
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
            material.scatter(&ray, &hit).unwrap();
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
            material.scatter(&ray, &hit).unwrap();
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
            material.scatter(&ray, &hit).unwrap();
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
            material.scatter(&ray, &hit).unwrap();
        });
    }
}