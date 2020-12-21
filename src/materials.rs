
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
    pub odd: Vec3,
    pub even: Vec3,
}

impl Texture for CheckerTexture {
    fn value(&self, _u: f32, _v: f32, p: Vec3) -> Vec3 {
        let sines = (self.check_size * p).map(|x| x.sin());
        let sines = sines.x * sines.y * sines.z;
        if sines < 0.0 {
            self.odd
        }
        else {
            self.even
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

pub enum Material {
    Invisible,
    Lambertian {
        albedo: Vec3
    },
    TexturedLambertian {
        albedo: Box<dyn Texture>
    },
    Metal {
        albedo: Vec3,
        fuzz: f32,
    },
    PolishedStone {
        albedo: Box<dyn Texture>,
    },
    Dielectric {
        ref_index: f32,
    },
    DiffuseLight {
        emission_colour: Vec3,
    }
}

pub fn scatter(ray: &Ray, hit: &HitRecord) -> Option<ScatterResult> {
    match hit.material {
        Material::Invisible => None,
        Material::Lambertian { albedo } =>
            lambertian_scatter(*albedo, hit.p, hit.normal),
        Material::TexturedLambertian { albedo } =>
            lambertian_scatter(albedo.value(0.0, 0.0, hit.p), hit.p, hit.normal),
        Material::Metal { albedo, fuzz } =>
            metal_scatter(*albedo, *fuzz, ray.direction, hit.p, hit.normal),
        Material::PolishedStone { albedo } =>
            polished_stone_scatter(&**albedo, ray.direction, hit.p, hit.normal),
        Material::Dielectric { ref_index } =>
            dielectric_scatter(*ref_index, ray.direction, hit.p, hit.normal),
        Material::DiffuseLight { emission_colour: _ } => None,
    }
}

pub fn emit(material: &Material) -> Vec3 {
    match material {
        Material::DiffuseLight { ref emission_colour } => *emission_colour,
        _ => Vec3::new(0.0, 0.0, 0.0)
    }
}

fn lambertian_scatter(albedo: Vec3, p: Vec3, normal: Vec3) -> Option<ScatterResult> {
    let target = p + normal + Vec3::random_in_unit_sphere(&mut rand::thread_rng());
    let dir = target - p;
    let attenuation = albedo;
    let scattered = Ray { origin: p, direction: dir };
    Some(ScatterResult { attenuation, scattered})
}

fn metal_scatter(albedo: Vec3, fuzz: f32, ray_dir: Vec3, p: Vec3, normal: Vec3) -> Option<ScatterResult> {
    let reflected = reflect(ray_dir.normalise(), normal);
    let reflected = reflected + fuzz * Vec3::random_in_unit_sphere(&mut rand::thread_rng());
    if dot(reflected, normal) > 0.0 {
        let scattered = Ray { origin: p, direction: reflected };
        Some(ScatterResult { attenuation: albedo, scattered})
    }
    else {
        None
    }
}

fn polished_stone_scatter(albedo: &dyn Texture, ray_dir: Vec3, p: Vec3, normal: Vec3) -> Option<ScatterResult> {
    let mut rng = rand::thread_rng();
    let reflected = reflect(ray_dir.normalise(), normal);
    let dotty = dot(reflected, normal);
    let reflect_prob = 1.0 - dotty.sqrt();
    let (attenuation, direction) = if rng.gen::<f32>() < reflect_prob {
        (Vec3::new(1.0, 1.0, 1.0), reflected)
    }
    else {
        let attenuation = albedo.value(0.0, 0.0, p);
        let scattered = normal + Vec3::random_in_unit_sphere(&mut rng);
        (attenuation, scattered)
    };
    let scattered = Ray { origin: p, direction };
    Some(ScatterResult { attenuation, scattered})
}

fn schlick(cosine: f32, ref_index: f32) -> f32 {
    let r0 = (1.0 - ref_index) / (1.0 + ref_index);
    let r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

fn dielectric_scatter(ref_index: f32, ray_dir: Vec3, p: Vec3, normal: Vec3) -> Option<ScatterResult> {
    let dir = ray_dir.normalise();
    let reflected = reflect(dir, normal);
    let attenuation = Vec3::new(1.0, 1.0, 1.0);

    let ray_dot_norm = dot(ray_dir, normal);
    let cosine = ray_dot_norm / ray_dir.len_sq().sqrt();
    let (outward_normal, ni_over_nt, cosine) =
        if ray_dot_norm > 0.0 {
            // Exiting the material
            (-normal, ref_index, ref_index * cosine)
        }
        else {
            // Entering the material
            (normal, 1.0 / ref_index, -cosine)
        };
    
    let (refracted, reflect_prob) =
        match refract(dir, outward_normal, ni_over_nt) {
            Some(refracted) => {
                (refracted, schlick(cosine, ref_index))
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
    let scattered = Ray { origin: p, direction: ray_dir };
    Some(ScatterResult { attenuation, scattered })
}

#[cfg(test)]
mod tests {
    use super::super::geometry::HitRecord;
    use super::*;

    #[test]
    fn scatter_lambertian() {
        // Given:
        let red = Vec3::new(1.0, 0.0, 0.0);
        let material = Material::Lambertian { albedo: red };
        let ray = Ray { origin: Vec3::new(1.0, 1.0, 0.0), direction: Vec3::new(-1.0, 0.0, 0.0) };
        let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

        // When: We scatter off the material
        let res = scatter(&ray, &hit);

        // Then: the result is not None
        let res = res.unwrap();

        // Then: the attenuation is the red that we defined on the material
        assert_eq!(res.attenuation, red);
    }

    #[test]
    fn scatter_metal() {
        // Given:
        let material = Material::Metal { albedo: Vec3::new(1.0, 0.0, 0.0), fuzz: 0.0 };
        let ray = Ray { origin: Vec3::new(-1.0, 2.0, 0.0), direction: Vec3::new(1.0, -1.0, 0.0) };
        let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

        // When: We scatter off the material
        let res = scatter(&ray, &hit);

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
        let material = Material::Dielectric { ref_index: 1.5 };
        let ray = Ray { origin: Vec3::new(-1.0, 2.0, 0.0), direction: Vec3::new(1.0, -1.0, 0.0) };
        let hit = HitRecord { t: 0.5, p: Vec3::new(0.0, 1.0, 1.0), normal: Vec3::new(0.0, 1.0, 0.0), material: &material };

        // When: We scatter off the material
        let res = scatter(&ray, &hit);

        // Then: the result is not None
        let res = res.unwrap();

        // Then: the attenuation is the red that we defined on the material
        assert_eq!(res.attenuation, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(res.scattered.origin, hit.p);
        assert_eq!(res.scattered.direction, Vec3::new(0.47140455, -0.8819171, 0.0)); // FIXME This depends on a random factor
    }
}