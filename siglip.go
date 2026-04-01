package siglip

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	imgSize    = 256
	embDim     = 768
	maxTextLen = 64

	meanR, meanG, meanB = 0.5, 0.5, 0.5
	stdR, stdG, stdB    = 0.5, 0.5, 0.5

	logitScaleInit = 4.709494113922119
	logitBias      = -12.885268211364746
)

var requiredFiles = []string{
	"vision_model.onnx",
	"text_model.onnx",
	"vocab.json",
}

var (
	ortMu       sync.Mutex
	ortRefCount int
)

// Result holds the classification score for a single label.
type Result struct {
	Label  string
	Score  float64 // sigmoid probability [0, 1]
	Cosine float64 // raw cosine similarity
}

// SearchResult holds the result of matching a precomputed embedding against a label.
type SearchResult struct {
	Index  int     // original index in the embeddings slice
	Score  float64 // sigmoid probability [0, 1]
	Cosine float64 // raw cosine similarity
}

// Classifier performs zero-shot image classification with SigLIP.
type Classifier struct {
	tok        *tokenizer
	modelDir   string
	managesORT bool
}

type config struct {
	modelDir     string
	runtimeLib   string
	autoDownload string // HuggingFace repo ID, empty = use default
	noDownload   bool   // disable auto-download
	skipORTInit  bool
}

// Option configures a Classifier.
type Option func(*config)

// WithModelDir sets the directory containing ONNX model files and vocab.json.
// Defaults to ~/.cache/siglip-go/.
func WithModelDir(dir string) Option {
	return func(c *config) { c.modelDir = dir }
}

// WithRuntimeLib sets the path to the ONNX Runtime shared library.
// If not set, common system paths are searched automatically.
func WithRuntimeLib(path string) Option {
	return func(c *config) { c.runtimeLib = path }
}

// WithAutoDownload overrides the default HuggingFace repository for model
// downloads. By default, models are downloaded from the built-in repo.
func WithAutoDownload(hfRepoID string) Option {
	return func(c *config) { c.autoDownload = hfRepoID }
}

// WithNoDownload disables automatic download of models and ONNX Runtime.
// Use this if you want to manage model files yourself.
func WithNoDownload() Option {
	return func(c *config) { c.noDownload = true }
}

// WithSkipORTInit skips ONNX Runtime initialization. Use this if your
// application already manages the ONNX Runtime lifecycle.
func WithSkipORTInit() Option {
	return func(c *config) { c.skipORTInit = true }
}

// New creates a new Classifier. By default, models and ONNX Runtime are
// downloaded automatically to ~/.cache/siglip-go/ on first use.
func New(opts ...Option) (*Classifier, error) {
	cfg := &config{
		modelDir: defaultCacheDir(),
	}
	for _, o := range opts {
		o(cfg)
	}

	// Auto-download models unless disabled
	if !cfg.noDownload {
		repo := cfg.autoDownload
		if repo == "" {
			repo = defaultHFRepo
		}
		if err := ensureModels(cfg.modelDir, repo); err != nil {
			return nil, fmt.Errorf("siglip: downloading models: %w", err)
		}
	}

	for _, f := range requiredFiles {
		p := filepath.Join(cfg.modelDir, f)
		if _, err := os.Stat(p); err != nil {
			return nil, fmt.Errorf("siglip: model file not found: %s\n"+
				"  Run siglip.New() with network access to auto-download, or\n"+
				"  use WithModelDir() to point to existing model files", p)
		}
	}

	ownORT := false
	if !cfg.skipORTInit {
		if err := acquireORT(cfg); err != nil {
			return nil, err
		}
		ownORT = true
	}

	tok, err := loadSigLIPTokenizer(filepath.Join(cfg.modelDir, "vocab.json"))
	if err != nil {
		if ownORT {
			releaseORT()
		}
		return nil, fmt.Errorf("siglip: loading tokenizer: %w", err)
	}

	return &Classifier{tok: tok, modelDir: cfg.modelDir, managesORT: ownORT}, nil
}

// Close releases resources held by the Classifier.
// If this classifier owns the last ORT reference, it also tears down the
// shared ONNX Runtime environment. Classifiers created with
// [WithSkipORTInit] never touch the ORT lifecycle.
func (c *Classifier) Close() {
	if c.managesORT {
		releaseORT()
		c.managesORT = false
	}
}

// acquireORT initializes the ONNX Runtime environment (on first call) and
// increments the reference count.
func acquireORT(cfg *config) error {
	ortMu.Lock()
	defer ortMu.Unlock()

	if ortRefCount == 0 {
		lib := cfg.runtimeLib
		if lib == "" {
			if !cfg.noDownload {
				var err error
				lib, err = ensureORT(cfg.modelDir)
				if err != nil {
					return err
				}
			} else {
				lib = findRuntimeLib()
			}
		}
		if lib == "" {
			return fmt.Errorf("siglip: ONNX Runtime library not found; " +
				"install it or use WithRuntimeLib()")
		}
		ort.SetSharedLibraryPath(lib)
		if err := ort.InitializeEnvironment(); err != nil {
			return err
		}
	}
	ortRefCount++
	return nil
}

// releaseORT decrements the ORT reference count and destroys the environment
// when no more classifiers need it.
func releaseORT() {
	ortMu.Lock()
	defer ortMu.Unlock()

	ortRefCount--
	if ortRefCount <= 0 {
		ort.DestroyEnvironment()
		ortRefCount = 0
	}
}

// ImageEmbedding returns the normalized 768-dimensional embedding for an image.
func (c *Classifier) ImageEmbedding(imagePath string) ([]float32, error) {
	return c.imageEmbedding(imagePath)
}

// Classify returns classification scores for the image against each label,
// sorted by score descending. Labels can be in any language.
func (c *Classifier) Classify(imagePath string, labels []string) ([]Result, error) {
	imgEmbed, err := c.imageEmbedding(imagePath)
	if err != nil {
		return nil, fmt.Errorf("siglip: image embedding: %w", err)
	}

	logitScale := math.Exp(logitScaleInit)
	results := make([]Result, len(labels))

	for i, label := range labels {
		textEmbed, err := c.textEmbedding(label)
		if err != nil {
			return nil, fmt.Errorf("siglip: text embedding for %q: %w", label, err)
		}
		cos := CosineSimilarity(imgEmbed, textEmbed)
		logit := cos*logitScale + logitBias
		results[i] = Result{
			Label:  label,
			Score:  Sigmoid(logit),
			Cosine: cos,
		}
	}

	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	return results, nil
}

// Search scores a list of precomputed image embeddings against a single text
// label and returns the top-k matches sorted by score descending.
// Each embedding must be a normalized 768-dimensional vector (as returned by
// ImageEmbedding). If k is larger than len(embeddings), all results are returned.
func (c *Classifier) Search(embeddings [][]float32, label string, k int) ([]SearchResult, error) {
	textEmbed, err := c.textEmbedding(label)
	if err != nil {
		return nil, fmt.Errorf("siglip: text embedding for %q: %w", label, err)
	}

	logitScale := math.Exp(logitScaleInit)
	matches := make([]SearchResult, len(embeddings))

	for i, emb := range embeddings {
		cos := CosineSimilarity(emb, textEmbed)
		logit := cos*logitScale + logitBias
		matches[i] = SearchResult{
			Index:  i,
			Score:  Sigmoid(logit),
			Cosine: cos,
		}
	}

	sort.Slice(matches, func(i, j int) bool { return matches[i].Score > matches[j].Score })

	if k > len(matches) {
		k = len(matches)
	}
	return matches[:k], nil
}

func defaultCacheDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "siglip-go")
}

func findRuntimeLib() string {
	var candidates []string

	// Check cache dir first
	cacheLib := filepath.Join(defaultCacheDir(), ortLibName())
	candidates = append(candidates, cacheLib)

	switch runtime.GOOS {
	case "darwin":
		candidates = append(candidates,
			"./libonnxruntime.dylib",
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
		)
	case "linux":
		candidates = append(candidates,
			"./libonnxruntime.so",
			"/usr/lib/libonnxruntime.so",
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
		)
	case "windows":
		candidates = append(candidates,
			"./onnxruntime.dll",
		)
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}
