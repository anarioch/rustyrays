pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {

    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }
    pub fn clone(&self) -> Vec3 {
        Vec3 { x: self.x, y: self.y, z: self.z }
    }

    pub fn normalise(&self) -> Vec3 {
        let length = self.dot(&self).sqrt();
        self.mul(1.0 / length)
    }

    // TODO: Figure out how to make these operator overloads
    pub fn add(&self, other: &Vec3) -> Vec3 {
        Vec3 { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    pub fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3 { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    pub fn mul(&self, scale: f32) -> Vec3 {
        Vec3 { x: self.x * scale, y: self.y * scale, z: self.z * scale }
    }

    pub fn dot(&self, other: &Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn len_sq(&self) -> f32 {
        self.dot(&self)
    }
}

pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: &Vec3, direction: &Vec3) -> Ray {
        Ray { origin: origin.clone(), direction: direction.clone() }
    }
    pub fn at_t(&self, t: f32) -> Vec3 {
        self.origin.add(&self.direction.mul(t))
    }
}

pub fn hit_sphere(centre: &Vec3, radius: f32, ray: &Ray) -> bool {
    let oc = ray.origin.sub(&centre);
    let a = ray.direction.len_sq();
    let b = 2.0 * oc.dot(&ray.direction);
    let c = oc.len_sq() - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    discriminant > 0.0
}

#[cfg(test)]
mod tests {
    use super::Vec3;
    use super::Ray;
    use super::hit_sphere;

    #[test]
    fn normalise() {
        // Normalise an already normalised vector
        let up = Vec3::new(0.0, 1.0, 0.0);
        let normalised = up.normalise();
        assert_eq!(normalised.x, 0.0);
        assert_eq!(normalised.y, 1.0);
        assert_eq!(normalised.z, 0.0);

        // Normalise a longer vector
        let up = Vec3::new(0.0, 3.0, 0.0);
        let normalised = up.normalise();
        assert_eq!(normalised.x, 0.0);
        assert_eq!(normalised.y, 1.0);
        assert_eq!(normalised.z, 0.0);
    }

    #[test]
    fn hit_sphere_works() {
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let left = Vec3::new(-1.0, 0.0, 0.0);
        let down_z = Ray { origin: origin.clone(), direction: Vec3::new(0.0, -1.0, 0.0) };
        let down_z_parallel = Ray { origin: left.mul(2.0), direction: Vec3::new(0.0, -1.0, 0.0) };
        assert_eq!(hit_sphere(&origin, 1.0, &down_z), true);
        assert_eq!(hit_sphere(&origin, 1.0, &down_z_parallel), false);
    }
}