use crate::conv_block::SuperPointConvBlock;
use candle_core::{Device, Module, Tensor, Result};

pub struct SuperPointKeypointDecoder {
    conv_blocks: Vec<SuperPointConvBlock>,
}

impl SuperPointKeypointDecoder {
    pub fn new(mut hidden_sizes: Vec<usize>, device: &Device) -> Result<Self> {
        if hidden_sizes.len() % 2 != 1 {
            Err("hidden_size does not contain an impair number of element")
        }
        let input_dim = 1;
        hidden_sizes.reverse();
        let first_hidden_size = *hidden_sizes.get(0)?;
        let mut conv_blocks = vec![];
        conv_blocks.push(SuperPointConvBlock::new(
            input_dim,
            first_hidden_size,
            true,
            device,
        )?);
        for i in 1..(hidden_sizes.len() - 1) {
            conv_blocks.push(SuperPointConvBlock::new(
                *hidden_sizes.get(i - 1)?,
                *hidden_sizes.get(i)?,
                true,
                device,
            )?)
        }
        conv_blocks.push(SuperPointConvBlock::new(
            *hidden_sizes.get(-2)?,
            *hidden_sizes.get(-1)?,
            true,
            device,
        )?);
        Ok(Self {
            conv_blocks,
        })
    }
}

impl Module for SuperPointKeypointDecoder {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let conv: Tensor;
        for conv_block in self.conv_blocks {
            conv = conv_block.forward(xs)?;
        }
    }
}