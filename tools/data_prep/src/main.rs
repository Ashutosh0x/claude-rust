use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use clap::Parser;

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    input: PathBuf,
    #[arg(short, long)]
    output_dir: PathBuf,
    #[arg(short, long, default_value_t = 1000)]
    lines_per_shard: usize,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    
    if !cli.output_dir.exists() {
        std::fs::create_dir_all(&cli.output_dir)?;
    }
    
    let file = File::open(&cli.input)?;
    let reader = BufReader::new(file);
    
    let mut shard_idx = 0;
    let mut line_count = 0;
    let mut writer = None;
    
    for line in reader.lines() {
        let line = line?;
        if line_count % cli.lines_per_shard == 0 {
            let shard_path = cli.output_dir.join(format!("shard_{:04}.txt", shard_idx));
            println!("Creating shard: {:?}", shard_path);
            writer = Some(File::create(shard_path)?);
            shard_idx += 1;
        }
        
        if let Some(ref mut w) = writer {
            writeln!(w, "{}", line)?;
        }
        line_count += 1;
    }
    
    println!("Done. Created {} shards from {} lines.", shard_idx, line_count);
    Ok(())
}
