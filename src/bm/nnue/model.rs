use std::io::Read;
use std::fmt::Debug;

use super::layers::{BitLinear, Linear};

// Include arch data
include!(concat!(env!("OUT_DIR"), "/arch.rs"));

macro_rules! read_n {
    ($src:expr, $type:ty) => {{
        let mut buffer = <$type>::to_le_bytes(0);
        $src.read_exact(&mut buffer)
            .map(|_| <$type>::from_le_bytes(buffer))
    }}
}

pub struct Nnue {
    pub ft: Box<BitLinear<INPUT, MID>>,
    pub l1: Linear<{ MID * 2 }, OUTPUT>
}

impl Debug for Nnue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Nnue {{ ft: _, l1: _ }}")
    }
}

impl Nnue {
    pub fn read_from(src: &mut impl Read) -> std::io::Result<Self> {
        let input_size = read_n!(src, u32)? as usize;
        let mid_size = read_n!(src, u32)? as usize;
        let out_size = read_n!(src, u32)? as usize;
        if input_size != INPUT || mid_size != MID || out_size != out_size {
            return Err(std::io::ErrorKind::InvalidData.into());
        }

        let ft = read_ft(src)?;
        let l1 = read_linear(src)?;
        Ok(Self { ft, l1 })
    }
}

fn read_ft<const INPUTS: usize, const OUTPUTS: usize>(src: &mut impl Read) -> std::io::Result<Box<BitLinear<INPUTS, OUTPUTS>>> {
    let mut ft = bytemuck::zeroed_box::<BitLinear<INPUTS, OUTPUTS>>();
    for weight in ft.weights.iter_mut().flatten() {
        *weight = read_n!(src, i16)?;
    }
    for bias in ft.biases.iter_mut() {
        *bias = read_n!(src, i16)?;
    }
    Ok(ft)
}

fn read_linear<const INPUTS: usize, const OUTPUTS: usize>(src: &mut impl Read) -> std::io::Result<Linear<INPUTS, OUTPUTS>> {
    let mut linear = Linear {
        weights: [[0; INPUTS]; OUTPUTS],
        biases: [0; OUTPUTS],
    };
    for weight in linear.weights.iter_mut().flatten() {
        *weight = read_n!(src, i8)?;
    }
    for bias in linear.biases.iter_mut() {
        *bias = read_n!(src, i16)? as i32;
    }
    Ok(linear)
}
