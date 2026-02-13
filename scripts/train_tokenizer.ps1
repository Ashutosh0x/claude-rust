# Train Tokenizer and prepare data
$env:CARGO_TARGET_DIR = 'C:\temp\rust-target'
$CORPUS = "data/raw/sample_corpus.txt"
$OUT_DIR = "data/tokenizer"

if (!(Test-Path $OUT_DIR)) {
    New-Item -ItemType Directory -Path $OUT_DIR
}

Write-Host "--- Training Tokenizer ---"
& "c:\Users\ashut\.cargo\bin\cargo.exe" run -p tokenizer_cli -- train $CORPUS --output-dir $OUT_DIR --vocab-size 5000

Write-Host "--- Copying Vocab to root data/ for TUI/Inference ---"
Copy-Item "$OUT_DIR/vocab.json" "data/vocab.json" -Force

Write-Host "Tokenizer ready."
