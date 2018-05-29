
#[derive(PartialEq,Debug)]
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

    pub fn set(&mut self, x: f32, y: f32, z: f32) {
        self.x = x;
        self.y = y;
        self.z = z;
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

    pub fn add_eq(&mut self, other: &Vec3) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }

    pub fn sub_eq(&mut self, other: &Vec3) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }

    pub fn mul_eq(&mut self, scale: f32) {
        self.x *= scale;
        self.y *= scale;
        self.z *= scale;
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

#[cfg(test)]
mod tests {
    use super::Vec3;
    use super::Ray;

    #[test]
    fn vec3_add() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(-1.0, 5.0, 0.0);

        assert_eq!(v1.add(&v2), Vec3::new(0.0, 7.0, 3.0));
    }

    #[test]
    fn vec3_sub() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(-1.0, 5.0, 0.0);

        assert_eq!(v1.sub(&v2), Vec3::new(2.0, -3.0, 3.0));
    }

    #[test]
    fn vec3_len_sq() {
        assert_eq!(Vec3::new(0.0, 0.0, 0.0).len_sq(), 0.0);
        assert_eq!(Vec3::new(0.0, 1.0, 0.0).len_sq(), 1.0);
        assert_eq!(Vec3::new(0.0, 5.0, 0.0).len_sq(), 25.0);
        assert_eq!(Vec3::new(1.0, 1.0, 1.0).len_sq(), 3.0);
    }

    #[test]
    fn vec3_mul() {
        let v1 = Vec3::new(1.0, 2.0, -3.0);

        assert_eq!(v1.mul(3.0), Vec3::new(3.0, 6.0, -9.0));
    }

    #[test]
    fn vec3_normalise() {
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
}
