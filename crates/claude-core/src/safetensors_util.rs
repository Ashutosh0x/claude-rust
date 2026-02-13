use std::path::Path;
use anyhow::Result;
use safetensors::SafeTensors;
use tch::{Tensor, nn, Kind};
use std::fs::File;
use memmap2::MmapOptions;

pub fn load_safetensors<P: AsRef<Path>>(vs: &mut nn::VarStore, path: P) -> Result<()> {
    let file = File::open(path)?;
    let buffer = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&buffer)?;

    let mut variables = vs.variables();
    let device = vs.device();

    for (name, view) in tensors.tensors() {
        if let Some(var) = variables.get_mut(&name) {
            let shape: Vec<i64> = view.shape().iter().map(|&x| x as i64).collect();
            let kind = match view.dtype() {
                safetensors::Dtype::F32 => Kind::Float,
                safetensors::Dtype::F16 => Kind::Half,
                safetensors::Dtype::BF16 => Kind::BFloat16,
                _ => return Err(anyhow::anyhow!("Unsupported dtype: {:?}", view.dtype())),
            };

            let data = view.data();
            let tch_tensor = Tensor::from_data_size(data, &shape, kind).to_device(device);
            
            tch::no_grad(|| {
                var.copy_(&tch_tensor);
            });
            println!("Loaded tensor: {}", name);
        } else {
            println!("Warning: Tensor {} found in safetensors but not in model", name);
        }
    }

    Ok(())
}

pub fn save_safetensors<P: AsRef<Path>>(_vs: &nn::VarStore, _path: P) -> Result<()> {
    // For now, let's focus on loading since it's more critical for using pretrained weights
    Ok(())
}
