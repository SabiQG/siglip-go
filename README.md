# siglip-go

Zero-shot image classification as a Go package. Uses [SigLIP](https://huggingface.co/google/siglip-base-patch16-256-multilingual) (Google) via ONNX Runtime. Multilingual — labels work in any language.

**Everything is auto-downloaded on first use** — models (~1.4GB) and ONNX Runtime (~34MB) are fetched to `~/.cache/siglip-go/` the first time you call `New()`.

```go
import siglip "github.com/SabiQG/siglip-go"

c, _ := siglip.New()
results, _ := c.Classify("photo.jpg", []string{"a dog", "a cat", "un avión"})
fmt.Println(results[0].Label, results[0].Score) // "a dog" 0.817
```

## Install

```bash
go get github.com/SabiQG/siglip-go
```

That's it. No Python, no manual downloads. The first call to `siglip.New()` will:

1. Download ONNX Runtime for your platform (macOS/Linux, amd64/arm64)
2. Download the SigLIP ONNX model files from HuggingFace

Both are cached in `~/.cache/siglip-go/` and reused on subsequent runs.

## API

### `siglip.New(opts ...Option) (*Classifier, error)`

Creates a classifier. With no options, everything is auto-downloaded.

| Option                   | Description                                                                  |
| ------------------------ | ---------------------------------------------------------------------------- |
| `WithModelDir(dir)`      | Use models from a custom directory instead of auto-downloading               |
| `WithRuntimeLib(path)`   | Use a specific ONNX Runtime library instead of auto-detecting/downloading    |
| `WithAutoDownload(repo)` | Override the default HuggingFace repo for model downloads                    |
| `WithNoDownload()`       | Disable all auto-downloading (models + ORT). You must provide files manually |
| `WithSkipORTInit()`      | Skip ONNX Runtime initialization (if your app already manages it)            |

### `(*Classifier).Classify(imagePath string, labels []string) ([]Result, error)`

Classifies an image against the given labels. Returns results sorted by score (highest first).

```go
type Result struct {
    Label  string
    Score  float64 // sigmoid probability [0, 1]
    Cosine float64 // raw cosine similarity
}
```

Each label gets an independent sigmoid score (they don't sum to 100%).

### `(*Classifier).ImageEmbedding(imagePath string) ([]float32, error)`

Returns the normalized 768-dimensional embedding vector for an image.

### `(*Classifier).Close()`

Releases classifier resources. Does not destroy the global ONNX Runtime environment.

## Example

```go
package main

import (
    "fmt"
    "log"

    siglip "github.com/SabiQG/siglip-go"
)

func main() {
    c, err := siglip.New()
    if err != nil {
        log.Fatal(err)
    }
    defer c.Close()

    results, err := c.Classify("upload.jpg", []string{
        "a photo of food",
        "a photo of a person",
        "a landscape photo",
        "a document or screenshot",
    })
    if err != nil {
        log.Fatal(err)
    }

    for _, r := range results {
        fmt.Printf("  %-30s  %.1f%%\n", r.Label, r.Score*100)
    }
}
```

## Advanced: Manual Setup

If you prefer not to use auto-download, use `WithNoDownload()` and provide files manually.

### Model files

Generate models with the included Python scripts (requires `torch`, `transformers`, `sentencepiece`):

```bash
python tools/export_siglip.py    # → siglip_onnx/vision_model.onnx, text_model.onnx
python tools/export_vocab.py     # → siglip_onnx/vocab.json
```

Then point to them:

```go
c, err := siglip.New(
    siglip.WithNoDownload(),
    siglip.WithModelDir("./siglip_onnx"),
)
```

### ONNX Runtime

If the library is already installed system-wide (e.g. via Homebrew or in `/usr/local/lib`), it's auto-detected. Otherwise specify a custom path:

```go
c, err := siglip.New(
    siglip.WithRuntimeLib("/path/to/libonnxruntime.dylib"),
)
```

## CLI

A demo CLI is included in `cmd/classify/`:

```bash
go run ./cmd/classify photo.jpg
go run ./cmd/classify photo.jpg "dog,cat,bird,car"
```

## Project Structure

```
.
├── siglip.go           # Public API: New(), Classify(), Options
├── download.go         # Auto-download models + ORT runtime
├── image.go            # Image preprocessing + vision ONNX encoder
├── text.go             # Text ONNX encoder
├── tokenizer.go        # SentencePiece Unigram tokenizer (Viterbi)
├── vecmath.go          # normalize, cosine similarity, sigmoid
├── cmd/classify/       # Example CLI
├── tools/export_siglip.py    # Python: export model to ONNX (optional)
└── tools/export_vocab.py     # Python: export vocab to JSON (optional)
```

## How It Works

```
Image ──→ Resize/Crop 256×256 ──→ Normalize ──→ Vision ONNX ──→ embed (768-dim)
                                                                       │
                                                                       ├─→ cosine × exp(scale) + bias ──→ sigmoid
                                                                       │
Text  ──→ SentencePiece tokenize ──→ Pad to 64 ──→ Text ONNX ──→ embed (768-dim)
```

- **Image**: Resize shortest side to 256, center crop, normalize to [-1,1] with mean=0.5/std=0.5
- **Text**: SentencePiece Unigram tokenization with Viterbi segmentation + byte fallback (250K multilingual vocab)
- **Scoring**: Sigmoid per label (independent scores), not softmax
