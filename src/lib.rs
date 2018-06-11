// Needed for using 'cargo bench', though I don't fully follow why
#![feature(test)]
extern crate test;

extern crate rand;
extern crate noise;

pub mod math;
pub mod ppm;
pub mod geometry;
pub mod materials;
