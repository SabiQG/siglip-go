// Package siglip provides zero-shot image classification using Google's
// SigLIP model (siglip-base-patch16-256-multilingual) via ONNX Runtime.
//
// Zero-shot means you can classify images with any set of labels — no
// training required. Labels can be in any language thanks to the
// multilingual tokenizer.
//
// # Quick Start
//
// Create a classifier and classify an image:
//
//	c, err := siglip.New()
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer c.Close()
//
//	results, err := c.Classify("photo.jpg", []string{"a dog", "a cat", "a car"})
//	for _, r := range results {
//	    fmt.Printf("%-20s  %.1f%%\n", r.Label, r.Score*100)
//	}
//
// On first use, New automatically downloads and caches the ONNX Runtime
// library (~34 MB) and SigLIP model files (~1.4 GB) to ~/.cache/siglip-go/.
//
// # Classify vs Search
//
// [Classifier.Classify] takes an image path and a set of labels, and
// returns a [Result] per label sorted by score.
//
// [Classifier.Search] takes precomputed image embeddings (from
// [Classifier.ImageEmbedding]) and a text query, returning the top-k
// matching images. Use this when you want to embed images once and
// search many times.
//
// # Scoring
//
// Each label receives an independent sigmoid score in [0, 1] — scores
// do not sum to 1. A score of 0.8 means the model is confident the
// image matches that label, regardless of other labels' scores.
//
// # Options
//
// Use [WithModelDir], [WithRuntimeLib], [WithNoDownload], [WithAutoDownload],
// and [WithSkipORTInit] to customize the classifier setup.
package siglip
