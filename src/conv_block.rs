use candle_core::{CudaDevice, Device, Module, Result, Tensor};
use candle_nn::{Activation, Conv2d, Conv2dConfig};

pub struct SuperPointConvBlock {
    conv_a: Conv2d,
    conv_b: Conv2d,
    relu: Activation,
    pool: bool,
}

impl SuperPointConvBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        add_pooling: bool,
        device: &Device,
    ) -> Result<Self> {
        let conv_a_weight = Tensor::randn(0f32, 1., (out_channels, in_channels, 3, 3), device)?;
        let conv_a_bias = Some(Tensor::randn(0f32, 1., (out_channels,), device)?);
        let conv_b_weight = Tensor::randn(0f32, 1., (out_channels, out_channels, 3, 3), device)?;
        let conv_b_bias = Some(Tensor::randn(0f32, 1., (out_channels,), device)?);
        let conv_config = Conv2dConfig {
            padding: 1,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        Ok(Self {
            conv_a: Conv2d::new(conv_a_weight, conv_a_bias, conv_config.clone()),
            conv_b: Conv2d::new(conv_b_weight, conv_b_bias, conv_config.clone()),
            relu: Activation::Relu,
            pool: add_pooling,
        })
    }
}

impl Module for SuperPointConvBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let conv1 = self.conv_a.forward(xs)?;
        let activated_conv1 = self.relu.forward(&conv1)?;
        let conv2 = self.conv_b.forward(&activated_conv1)?;
        let activated_conv2 = self.relu.forward(&conv2)?;
        if self.pool {
            Ok(activated_conv2.max_pool2d_with_stride(2, 2)?)
        } else {
            Ok(activated_conv2)
        }
    }
}
