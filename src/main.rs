mod superpoint;

use std::time::Instant;
use candle_ext::candle::nn::VarBuilder;
use candle_ext::candle::{Device, DType, Error, IndexOp, Module, Result, Tensor};
use candle_ext::candle::D::Minus1;
use candle_ext::candle::DType::F32;
use hf_hub::api::sync::Api;
use crate::superpoint::{SuperPoint, SuperPointConfig};
use image::{GrayImage};


fn read_gray_image(image_path: &str, device: &Device) -> Result<Tensor> {
    let img = image::io::Reader::open(image_path)?
        .decode()
        .map_err(Error::wrap)?
        .resize_to_fill(640, 480, image::imageops::FilterType::Triangle);
    let gray_image = GrayImage::from(img).into_raw();

    let data = Tensor::from_vec(gray_image, (480, 640, 1), device)?.permute((2, 0, 1))?;
    data.to_dtype(F32)? / 255.
}

fn non_zero(t: &Tensor) -> Result<Tensor> {
    let (height, width) = t.dims2()?;

    let mut indices = vec![];
    let mut num = 0;
    for i in 0..height {
        for j in 0..width {
            let val: Tensor = t.i((i, j))?;
            let scalar: u32 = val.to_scalar()?;
            if scalar != 0 {
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

    let t = Tensor::new(&[[1u32, 2, 0, 5], [1, 0, 2, 2]], &t.device())?;
    let coordinates = non_zero(&t)?;

    let coordinates_t = coordinates.t()?;

    let coordinates_x = coordinates_t.i(0)?;
    let coordinates_y = coordinates_t.i(1)?;
    let t_x = t.index_select(&coordinates_x, 0)?;
    let t_y = t_x.index_select(&coordinates_y, 1)?;
    let (_, t_shape_minus_1) = t.dims2()?;
    let coordinates_shape_0 = coordinates_x.dims1()?;
    let b_coordinates_y = coordinates_y.unsqueeze(Minus1)?.broadcast_as((coordinates_shape_0, t_shape_minus_1))?;
    let gathered_t_x = t_x.gather(&b_coordinates_y.contiguous()?, Minus1)?;
    let max_gathered_t = gathered_t_x.max(Minus1)?;
    Ok(max_gathered_t.clone())
}

fn non_zero_with_values(device: &Device) -> Result<Tensor> {
    let t = Tensor::new(&[[1u32, 2, 0, 5], [1, 0, 2, 2]], device)?;
    let orig_shape = t.shape();

    let (h, w) = orig_shape.dims2()?;
    let x_range = Tensor::arange(0 as u32, h as u32, device)?.unsqueeze(Minus1)?.broadcast_as((h, w))?.flatten_all()?;
    let y_range = Tensor::arange(0 as u32, w as u32, device)?.repeat(h)?;
    let t_flat = t.flatten_all()?;
    Ok(t)
}

fn main() -> Result<()> {
    println!("Hello, world!");

    let device = Device::new_cuda(0)?;

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let c = a.matmul(&b)?;

    let text = Tensor::new(&[[1u32, 2, 0, 5], [1, 0, 2, 2]], &device)?;
    let text_nonzero = non_zero(&text)?;
    two_dim_index(&text, &text_nonzero)?;

    non_zero_with_values(&device)?;

    let t = Tensor::new(
        &[[[-0.6257f32, -2.0103, -0.0145],
            [0.7390, -3.5205, 0.5833],
            [-0.1699, 0.2690, 0.3833],
            [0.7064, -0.8789, 1.3285]],
            [[-1.0139, 0.9933, 1.4314],
                [-1.1443, -0.0604, 0.2624],
                [-0.2376, 2.1882, -1.2499],
                [-0.8305, 0.4660, 0.9780]]], &device)?;
    let text = Tensor::new(&[[1u32, 2, 0, 5], [1, 0, 2, 2]], &device)?;
    let am = text.argmax_keepdim(Minus1)?;
    let amu = am.unsqueeze(2)?;
    let amub = amu.broadcast_as((2, 1, 3))?;
    let result = t.gather(&amub.contiguous()?, 1)?;

    let t = Tensor::new(&[[2f32, 0., 0.], [0., 0., 0.], [0., 0., 3.]], &device)?;

    let api = Api::new().unwrap();
    let repo = api.model("stevenbucaille/superpoint".to_string());
    let model_file = repo.get("model.safetensors").unwrap();

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], F32, &device)? };
    let cfg = SuperPointConfig::default();
    let model = SuperPoint::new(vb, cfg)?;

    let image = read_gray_image("/home/steven/transformers_fork/transformers/tests/fixtures/tests_samples/COCO/000000004016.png", &device)?;
    println!("loaded image {image:?}");

    let now = Instant::now();
    let output = model.forward(&image.unsqueeze(0)?)?;
    println!("Time {}", now.elapsed().as_millis());
    println!("output : {}", output);

    Ok(())
}
