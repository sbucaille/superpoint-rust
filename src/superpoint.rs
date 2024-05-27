use candle_ext::candle::{IndexOp, Result, Tensor};
use candle_ext::candle::D::Minus1;
use candle_ext::candle::DType::F32;
use candle_ext::candle::nn::{Activation, conv2d, Conv2d, Conv2dConfig, Module, VarBuilder};
use candle_ext::candle::nn::ops::softmax;
use candle_ext::TensorExt;

fn logical_and(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    return x.logical_not()?.logical_or(&y.logical_not()?);
}

fn simple_nms(scores: &Tensor, nms_radius: usize) -> Result<Tensor> {
    fn max_pool(x: &Tensor, padding: usize) -> Result<Tensor> {
        let mut padded_x = x.pad_with_zeros(1, padding, padding)?.pad_with_zeros(2, padding, padding)?;
        padded_x = padded_x.unsqueeze(0)?;
        padded_x = padded_x.max_pool2d_with_stride(padding * 2 + 1, 1)?;
        padded_x = padded_x.squeeze(0)?;
        return Ok(padded_x);
    }

    let zeros = scores.zeros_like()?;
    let mut max_mask = scores.eq(&max_pool(scores, nms_radius)?)?;
    for _ in 0..2 {
        let max_mask_f32 = max_mask.to_dtype(F32)?;
        let supp_mask = max_pool(&max_mask_f32, nms_radius)?.gt(&zeros)?;
        let supp_scores = (scores * supp_mask.logical_not()?.to_dtype(F32))?;
        let new_max_mask = supp_scores.eq(&max_pool(&supp_scores, nms_radius)?)?;
        max_mask = max_mask.logical_or(&logical_and(&new_max_mask, &supp_mask.logical_not()?)?)?;
    }
    let max_mask_f32 = max_mask.to_dtype(F32)?;
    let result = (max_mask_f32 * scores)?;
    Ok(result)
}

fn top_k_keypoints(keypoints: &Tensor, scores: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let num_keypoints = keypoints.dims()[0];
    if k > num_keypoints {
        return Ok((keypoints.clone(), scores.clone()));
    }


    Ok((keypoints.clone(), scores.clone()))
}

fn non_zero_dims1(t: &Tensor) -> Result<Tensor> {
    // TODO generalize
    let vec_t = t.to_vec1::<u8>()?;
    let mut indices = vec![];
    let mut num = 0;
    for (i, &val) in vec_t.iter().enumerate() {
        if val != 0 {
            indices.push(i as u32);
            num += 1;
        }
    }

    let non_zero_indices = Tensor::from_vec(indices, num, &t.device())?;
    Ok(non_zero_indices)
}

fn non_zero_dims2(t: &Tensor) -> Result<Tensor> {
    let vec_t = t.to_vec2::<f32>()?;
    let mut indices = vec![];
    let mut num = 0;
    for (i, vec) in vec_t.iter().enumerate() {
        for (j, &val) in vec.iter().enumerate() {
            if val != 0. {
                indices.push(i as u32);
                indices.push(j as u32);
                num += 1;
            }
        }
    }

    let non_zero_indices = Tensor::from_vec(indices, (num, 2), &t.device())?;
    Ok(non_zero_indices)
}

fn two_dim_index(t: &Tensor, coordinates: &Tensor) -> Result<Tensor> {
    let coordinates_t = coordinates.t()?;

    let coordinates_x = coordinates_t.i(0)?;
    let coordinates_y = coordinates_t.i(1)?;

    let t_x = t.index_select(&coordinates_x, 0)?;

    let (_, t_shape_minus_1) = t.dims2()?;
    let coordinates_shape_0 = coordinates_x.dims1()?;
    let b_coordinates_y = coordinates_y.unsqueeze(Minus1)?.broadcast_as((coordinates_shape_0, t_shape_minus_1))?;
    let gathered_t_x = t_x.gather(&b_coordinates_y.contiguous()?, Minus1)?;
    let max_gathered_t = gathered_t_x.max(Minus1)?;

    Ok(max_gathered_t.clone())
}

#[derive(Debug, Clone, Copy)]
pub struct SuperPointConfig {
    encoder_hidden_sizes: [usize; 4],
    decoder_hidden_size: usize,
    keypoint_decoder_dim: usize,
    descriptor_decoder_dim: usize,
    keypoint_threshold: f32,
    max_keypoints: usize,
    nms_radius: usize,
    border_removal_distance: i32,
}

impl SuperPointConfig {
    pub fn default() -> Self {
        Self {
            encoder_hidden_sizes: [64, 64, 128, 128],
            decoder_hidden_size: 256,
            keypoint_decoder_dim: 65,
            descriptor_decoder_dim: 256,
            keypoint_threshold: 0.005,
            max_keypoints: 512,
            nms_radius: 4,
            border_removal_distance: 4,
        }
    }
}

pub struct SuperPointConvBlock {
    conv_a: Conv2d,
    conv_b: Conv2d,
    relu: Activation,
    pool: bool,
}

impl SuperPointConvBlock {
    pub fn new(p: VarBuilder,
               in_channels: usize,
               out_channels: usize,
               add_pooling: bool) -> Result<Self> {
        let conv_config = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        let conv_a_weights = p.pp("conv_a");
        let conv_a = conv2d(in_channels, out_channels, 3, conv_config, conv_a_weights).unwrap();
        let conv_b_weights = p.pp("conv_b");
        let conv_b = conv2d(out_channels, out_channels, 3, conv_config, conv_b_weights).unwrap();

        Ok(Self {
            conv_a,
            conv_b,
            relu: Activation::Relu,
            pool: add_pooling,
        })
    }
}

impl Module for SuperPointConvBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        println!("{}", xs);
        let mut hidden_states = self.conv_a.forward(&xs)?;
        hidden_states = Activation::Relu.forward(&hidden_states)?;
        hidden_states = self.conv_b.forward(&hidden_states)?;
        hidden_states = Activation::Relu.forward(&hidden_states)?;

        if self.pool {
            hidden_states = hidden_states.max_pool2d_with_stride(2, 2)?;
        }
        Ok(hidden_states)
    }
}


pub struct SuperPointEncoder {
    conv_blocks: Vec<SuperPointConvBlock>,
}

impl SuperPointEncoder {
    pub fn new(p: VarBuilder, cfg: SuperPointConfig) -> Result<Self> {
        let hidden_sizes = cfg.encoder_hidden_sizes;

        // if hidden_sizes.len() % 2 != 1 {
        //     Error::Msg("hidden_size does not contain an impair number of element".to_string())
        // }
        let input_dim = 1;
        let first_hidden_size = hidden_sizes[0];
        let mut conv_blocks = vec![];

        let conv_0_weights = p.pp("conv_blocks.0");
        conv_blocks.push(SuperPointConvBlock::new(
            conv_0_weights,
            input_dim,
            first_hidden_size,
            true,
        )?);
        for i in 1..(hidden_sizes.len() - 1) {
            let conv_i_weights = p.pp(format!("conv_blocks.{}", i));
            conv_blocks.push(SuperPointConvBlock::new(
                conv_i_weights,
                hidden_sizes[i - 1],
                hidden_sizes[i],
                true,
            )?)
        }
        let conv_3_weights = p.pp("conv_blocks.3");
        conv_blocks.push(SuperPointConvBlock::new(
            conv_3_weights,
            hidden_sizes[2],
            hidden_sizes[3],
            false,
        )?);
        Ok(Self {
            conv_blocks,
        })
    }
}

impl Module for SuperPointEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut ys = xs.clone();
        for conv_block in self.conv_blocks.iter() {
            ys = conv_block.forward(&ys)?;
        }
        return Ok(ys);
    }
}

pub struct SuperPointKeypointDecoder {
    keypoint_threshold: f32,
    max_keypoints: usize,
    nms_radius: usize,
    border_removal_distance: i32,
    conv_score_a: Conv2d,
    conv_score_b: Conv2d,
}

impl SuperPointKeypointDecoder {
    pub fn new(p: VarBuilder, cfg: SuperPointConfig) -> Result<Self> {
        let conv_score_a_weights = p.pp("conv_score_a");
        let conv_score_a = conv2d(
            cfg.encoder_hidden_sizes[3],
            cfg.decoder_hidden_size,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            conv_score_a_weights,
        ).unwrap();

        let conv_score_b_weights = p.pp("conv_score_b");
        let conv_score_b = conv2d(
            cfg.decoder_hidden_size,
            cfg.keypoint_decoder_dim,
            1,
            Conv2dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
            },
            conv_score_b_weights,
        ).unwrap();

        Ok(Self {
            keypoint_threshold: cfg.keypoint_threshold,
            max_keypoints: cfg.max_keypoints,
            nms_radius: cfg.nms_radius,
            border_removal_distance: cfg.border_removal_distance,
            conv_score_a,
            conv_score_b,
        })
    }

    fn get_pixel_scores(&self, encoded_keypoints: &Tensor) -> Result<Tensor> {
        let mut scores = self.conv_score_a.forward(&encoded_keypoints)?;
        scores = Activation::Relu.forward(&scores)?;
        scores = self.conv_score_b.forward(&scores)?;
        scores = softmax(&scores, 1)?;
        let scores_dim = scores.dim(1)? - 1;
        scores = scores.i((.., ..scores_dim))?;
        let (batch_size, _, height, width) = scores.dims4()?;
        scores = scores.permute((0, 2, 3, 1))?.reshape((batch_size, height, width, 8, 8))?;
        scores = scores.permute((0, 1, 3, 2, 4))?.reshape((batch_size, height * 8, width * 8))?;
        scores = simple_nms(&scores, self.nms_radius)?;
        Ok(scores)
    }

    fn extract_keypoints(&self, mut scores: Tensor) -> Result<(Tensor, Tensor)> {
        scores = scores.i(0)?;
        let (height, width) = scores.dims2()?;

        let threshold = scores.full_like(self.keypoint_threshold)?;
        let threshold_scores = scores.gt(&threshold)?.to_dtype(F32)?;

        let mut keypoints = non_zero_dims2(&threshold_scores)?;
        scores = two_dim_index(&scores, &keypoints)?;

        (keypoints, scores) = remove_keypoints_from_borders(
            &keypoints, &scores, self.border_removal_distance, height * 8, width * 8,
        )?;

        if self.max_keypoints >= 0 {
            (keypoints, scores) = top_k_keypoints(
                &keypoints, &scores, self.max_keypoints,
            )?;
        }

        Ok((keypoints, scores))
    }
}

fn remove_keypoints_from_borders(keypoints: &Tensor, scores: &Tensor, border: i32, height: usize, width: usize) -> Result<(Tensor, Tensor)> {
    let keypoints_0 = keypoints.i((.., 0))?;
    let keypoints_1 = keypoints.i((.., 1))?;
    let border_t = keypoints_0.full_like(border as u32)?;
    let height_t = keypoints_0.full_like(height as u32)?;
    let width_t = keypoints_0.full_like(width as u32)?;
    let height_border = (&height_t - &border_t)?;
    let width_border = (&width_t - &border_t)?;
    let mask_h = logical_and(&keypoints_0.ge(&border_t)?, &keypoints_0.lt(&height_border)?)?;
    let mask_w = logical_and(&keypoints_1.ge(&border_t)?, &keypoints_1.lt(&width_border)?)?;
    let mask = logical_and(&mask_h, &mask_w)?;
    let mask_non_zero = non_zero_dims1(&mask)?;

    let keypoints_selected = keypoints.i(&mask_non_zero)?;
    let scores_selected = scores.i(&mask_non_zero)?;

    Ok((keypoints_selected, scores_selected))
}

impl Module for SuperPointKeypointDecoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut scores = self.get_pixel_scores(&xs)?;
        let (keypoints, scores) = self.extract_keypoints(scores)?;

        Ok(scores)
    }
}

pub struct SuperPointDescriptorDecoder {
    conv_descriptor_a: Conv2d,
    conv_descriptor_b: Conv2d,
}

impl SuperPointDescriptorDecoder {
    pub fn new(p: VarBuilder, cfg: SuperPointConfig) -> Result<Self> {
        let conv_descriptor_a_weights = p.pp("conv_descriptor_a");
        let conv_descriptor_a = conv2d(
            cfg.encoder_hidden_sizes[3],
            cfg.decoder_hidden_size,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 1,
                ..Default::default()
            },
            conv_descriptor_a_weights,
        ).unwrap();
        let conv_descriptor_b_weights = p.pp("conv_descriptor_b");
        let conv_descriptor_b = conv2d(
            cfg.decoder_hidden_size,
            cfg.descriptor_decoder_dim,
            1,
            Conv2dConfig {
                padding: 0,
                stride: 1,
                ..Default::default()
            },
            conv_descriptor_b_weights,
        ).unwrap();

        Ok(Self {
            conv_descriptor_a,
            conv_descriptor_b,
        })
    }
}

pub struct SuperPoint {
    encoder: SuperPointEncoder,
    keypoint_decoder: SuperPointKeypointDecoder,
    descriptor_decoder: SuperPointDescriptorDecoder,
}

impl SuperPoint {
    pub fn new(p: VarBuilder, cfg: SuperPointConfig) -> Result<Self> {
        let encoder_weights = p.pp("encoder");
        let encoder = SuperPointEncoder::new(encoder_weights, cfg).unwrap();
        let keypoint_decoder_weights = p.pp("keypoint_decoder");
        let keypoint_decoder = SuperPointKeypointDecoder::new(keypoint_decoder_weights, cfg).unwrap();
        let descriptor_decoder_weights = p.pp("descriptor_decoder");
        let descriptor_decoder = SuperPointDescriptorDecoder::new(descriptor_decoder_weights, cfg).unwrap();
        Ok(Self {
            encoder,
            keypoint_decoder,
            descriptor_decoder,
        })
    }
}

impl Module for SuperPoint {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let encoder_outputs = self.encoder.forward(xs)?;
        let keypoint_decoder_outputs = self.keypoint_decoder.forward(&encoder_outputs)?;
        Ok(keypoint_decoder_outputs)
    }
}