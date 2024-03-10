mod encoder;
mod conv_block;

use candle_core::{Device, Module, Result, Tensor};
//
// struct SuperPointConfig {}
//
// struct SuperPointEncoder {}
//
//
// struct SuperPointDescriptorDecoder {}
//
// struct SuperPoint {
//     config: SuperPointConfig,
//     encoder: SuperPointEncoder,
//     keypoint_decoder: SuperPointKeypointDecoder,
//     descriptor_decoder: SuperPointDescriptorDecoder,
// }
//
// impl SuperPoint {
//     pub fn new(config: SuperPointConfig, encoder: SuperPointEncoder, keypoint_decoder: SuperPointKeypointDecoder, descriptor_decoder: SuperPointDescriptorDecoder) -> Self {
//         Self {
//             config,
//             encoder,
//             keypoint_decoder,
//             descriptor_decoder,
//         }
//     }
//
//     pub fn config(&self) -> &SuperPointConfig {
//         &self.config
//     }
// }
//
// impl Module for SuperPoint {
//     fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
//         let encoded_keypoints_and_descriptors = self.keypoint_decoder.forward(pixel_values)?;
//     }
// }

fn main() -> Result<()> {
    println!("Hello, world!");

    let device = Device::new_cuda(0)?;

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let c = a.matmul(&b)?;
    println!("{c}");

    // Test SuperPointConvBlock
    let conv_block = crate::conv_block::SuperPointConvBlock::new(3, 128, false, &device)?;
    let x = Tensor::randn(0f32, 1., (1, 3, 640, 480), &device)?;
    let y = conv_block.forward(&x)?;
    println!("{}", y);

    // Test SuperPointKeypointEncoder

    Ok(())
}
