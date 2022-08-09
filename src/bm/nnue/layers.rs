use bytemuck::Zeroable;

const UNITS: i16 = 400;
const FT_SCALE: i16 = 255;
const SCALE: i16 = 64;
const MIN: i16 = 0;
const MAX: i16 = FT_SCALE;
const SHIFT: i16 = 8;

#[derive(Debug, Clone, Zeroable)]
pub struct BitLinear<const INPUT: usize, const OUTPUT: usize> {
    pub weights: [[i16; OUTPUT]; INPUT],
    pub biases: [i16; OUTPUT],
}

impl<const INPUT: usize, const OUTPUT: usize> BitLinear<INPUT, OUTPUT> {
    #[inline]
    pub fn modify_feature<const CHANGE: i16>(&self, index: usize, outputs: &mut [i16; OUTPUT]) {
        for (out, &weight) in outputs.iter_mut().zip(&self.weights[index]) {
            *out += weight * CHANGE;
        }
    }
}

#[derive(Debug, Clone, Zeroable)]
pub struct Linear<const INPUT: usize, const OUTPUT: usize> {
    pub weights: [[i8; INPUT]; OUTPUT],
    pub biases: [i32; OUTPUT],
}

impl<const INPUT: usize, const OUTPUT: usize> Linear<INPUT, OUTPUT> {
    #[inline]
    pub fn forward(&self, inputs: &[u8; INPUT]) -> [i32; OUTPUT] {
        let mut out = self.biases;
        for (out, weights) in out.iter_mut().zip(&self.weights) {
            for (&input, &weight) in inputs.iter().zip(weights.iter()) {
                *out += weight as i32 * input as i32;
            }
        }
        out
    }
}

#[inline]
pub fn out(x: i32) -> i16 {
    (x as f32 * UNITS as f32 / (FT_SCALE as f32 * SCALE as f32)) as i16
}

#[inline]
pub fn sq_clipped_relu<const N: usize>(input: &[i16; N], output: &mut [u8; N]) {
    for (&input, output) in input.iter().zip(output.iter_mut()) {
        let clamped = input.max(MIN).min(MAX) as u16;
        *output = ((clamped * clamped) >> SHIFT) as u8;
    }
}
