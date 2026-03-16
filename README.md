# siglip-go

[![Go Reference](https://pkg.go.dev/badge/github.com/SabiQG/siglip-go.svg)](https://pkg.go.dev/github.com/SabiQG/siglip-go)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Zero-shot image classification in pure Go. Powered by [SigLIP](https://huggingface.co/google/siglip-base-patch16-256-multilingual) (Google) and ONNX Runtime.

- **Zero dependencies** — no Python, no manual downloads
- **Multilingual** — labels work in any language
- **Auto-setup** — models and runtime are fetched on first use

```go
import siglip "github.com/SabiQG/siglip-go"

c, _ := siglip.New()
results, _ := c.Classify("photo.jpg", []string{"a dog", "a cat", "un avión"})
fmt.Println(results[0].Label, results[0].Score) // "a dog" 0.817
```

---

## Getting Started

```bash
go get github.com/SabiQG/siglip-go
```

The first call to `siglip.New()` automatically downloads and caches:

| Component    | Size    | Location              |
| ------------ | ------- | --------------------- |
| ONNX Runtime | ~34 MB  | `~/.cache/siglip-go/` |
| SigLIP model | ~1.4 GB | `~/.cache/siglip-go/` |

Subsequent runs reuse the cached files.

---

## Usage

### Classify an image

```go
c, err := siglip.New()
if err != nil {
    log.Fatal(err)
}
defer c.Close()

results, _ := c.Classify("photo.jpg", []string{
    "a photo of food",
    "a photo of a person",
    "a landscape photo",
})

for _, r := range results {
    fmt.Printf("%-30s  %.1f%%\n", r.Label, r.Score*100)
}
```

### Search across precomputed embeddings

Embed images once, then search by text as many times as needed:

```go
// Precompute embeddings
embs := make([][]float32, len(images))
for i, img := range images {
    embs[i], _ = c.ImageEmbedding(img)
}

// Find the 5 images that best match "a sunset"
matches, _ := c.Search(embs, "a sunset", 5)
for _, m := range matches {
    fmt.Printf("#%d  score=%.3f\n", m.Index, m.Score)
}
```

### CLI demo

```bash
go run ./cmd/classify photo.jpg
go run ./cmd/classify photo.jpg "dog,cat,bird,car"
```

---

## API Reference

### `siglip.New(opts ...Option) (*Classifier, error)`

Creates a classifier. With no options, everything is auto-downloaded.

**Options**

| Option                   | Description                                                               |
| ------------------------ | ------------------------------------------------------------------------- |
| `WithModelDir(dir)`      | Use models from a custom directory instead of auto-downloading            |
| `WithRuntimeLib(path)`   | Use a specific ONNX Runtime library instead of auto-detecting/downloading |
| `WithAutoDownload(repo)` | Override the default HuggingFace repo for model downloads                 |
| `WithNoDownload()`       | Disable all auto-downloading — you must provide files manually            |
| `WithSkipORTInit()`      | Skip ONNX Runtime initialization (if your app already manages it)         |

### `(*Classifier).Classify(imagePath string, labels []string) ([]Result, error)`

Classifies an image against the given labels. Returns results sorted by score descending.

Each label gets an **independent sigmoid score** — scores don't sum to 100%.

```go
type Result struct {
    Label  string
    Score  float64 // sigmoid probability [0, 1]
    Cosine float64 // raw cosine similarity
}
```

### `(*Classifier).ImageEmbedding(imagePath string) ([]float32, error)`

Returns the L2-normalized 768-dimensional embedding vector for an image.

### `(*Classifier).Search(embeddings [][]float32, label string, k int) ([]SearchResult, error)`

Scores precomputed image embeddings against a text label, returns the top-k matches sorted by score descending. If `k > len(embeddings)`, all results are returned.

```go
type SearchResult struct {
    Index  int     // position in the original embeddings slice
    Score  float64 // sigmoid probability [0, 1]
    Cosine float64 // raw cosine similarity
}
```

### `(*Classifier).Close()`

Releases classifier resources. Does not destroy the global ONNX Runtime environment.

---

## How It Works

```
Image ─→ Resize/Crop 256×256 ─→ Normalize ─→ vision_model.onnx ─→ embed (768-d)
                                                                        │
                                                                    cosine similarity
                                                                        │
Text  ─→ SentencePiece tokenize ─→ Pad to 64 ─→ text_model.onnx ─→ embed (768-d)

                        score = sigmoid(cosine × exp(scale) + bias)
```

| Stage | Details                                                                              |
| ----- | ------------------------------------------------------------------------------------ |
| Image | Resize shortest side to 256 → center crop → normalize to [-1, 1] (mean/std = 0.5)    |
| Text  | SentencePiece Unigram + Viterbi segmentation, byte fallback, 250K multilingual vocab |
| Score | Sigmoid per label (independent), not softmax                                         |

---

## Advanced

### Manual model setup

Use `WithNoDownload()` to skip auto-downloading and provide model files yourself:

```bash
python tools/export_siglip.py    # → vision_model.onnx, text_model.onnx
python tools/export_vocab.py     # → vocab.json
```

```go
c, _ := siglip.New(
    siglip.WithNoDownload(),
    siglip.WithModelDir("./siglip_onnx"),
)
```

### Custom ONNX Runtime path

System-wide installations (Homebrew, `/usr/local/lib`) are auto-detected. To specify manually:

```go
c, _ := siglip.New(
    siglip.WithRuntimeLib("/path/to/libonnxruntime.dylib"),
)
```

---

## Project Structure

```
├── siglip.go              Public API: New(), Classify(), Search(), Options
├── download.go            Auto-download models + ONNX Runtime
├── image.go               Image preprocessing + vision encoder
├── text.go                Text encoder
├── tokenizer.go           SentencePiece Unigram tokenizer (Viterbi)
├── vecmath.go             Normalize, cosine similarity, sigmoid
├── cmd/classify/          Demo CLI
└── tools/                 Python export scripts (optional)
```

---

## License

Apache 2.0
