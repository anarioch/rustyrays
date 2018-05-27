use ::math::Vec3;

pub struct PpmImage {
    width: usize,
    height: usize,
    body: String,
}

impl PpmImage {
    pub fn create(width: usize, height: usize) -> PpmImage {
        PpmImage { width, height, body: String::from("") }
    }

    pub fn append_pixel(&mut self, colour: &Vec3) {
        let colour = colour.mul(255.0);
        self.body.push_str(&format!("{:4} {:4} {:4}", colour.x as u32, colour.y as u32, colour.z as u32));
    }

    pub fn get_text(&self) -> String {
        let mut text = String::new();
        // COLS x ROWS; 255 is max colour
        text.push_str(&format!("P3\n{} {}\n255\n", self.width, self.height));
        text.push_str(&self.body);
        text
    }
}
