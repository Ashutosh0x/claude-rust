//! Weight initialization utilities for transformer models.

use tch::{nn, Tensor, Kind};

/// Initialize a linear layer with scaled normal distribution.
/// Standard transformer init: N(0, 0.02) for weights, zeros for bias.
pub fn init_linear(linear: &nn::Linear) {
    tch::no_grad(|| {
        let _ = linear.ws.init(nn::Init::Randn { mean: 0.0, stdev: 0.02 });
        if let Some(ref bs) = linear.bs {
            let _ = bs.init(nn::Init::Const(0.0));
        }
    });
}

/// GPT-2 style residual scaling: scale output projection by 1/sqrt(2*n_layer).
pub fn residual_scale(n_layer: i64) -> f64 {
    1.0 / (2.0 * n_layer as f64).sqrt()
}

/// Count total trainable parameters in a VarStore.
pub fn count_parameters(vs: &nn::VarStore) -> i64 {
    vs.variables()
        .iter()
        .map(|(_, t)| t.numel())
        .sum()
}

/// Print a summary of the model's parameter counts by layer.
pub fn print_param_summary(vs: &nn::VarStore) {
    let mut total = 0i64;
    println!("╔═══════════════════════════════════════════╗");
    println!("║           Model Parameter Summary         ║");
    println!("╠═══════════════════════════════════════════╣");
    for (name, tensor) in vs.variables() {
        let n = tensor.numel();
        total += n;
        println!("║ {:35} {:>8} ║", name, format_count(n));
    }
    println!("╠═══════════════════════════════════════════╣");
    println!("║ Total: {:>35} ║", format_count(total));
    println!("╚═══════════════════════════════════════════╝");
}

fn format_count(n: i64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}
