package siglip

import "math"

// Normalize applies L2 normalization to a vector in-place.
func Normalize(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	n := float32(math.Sqrt(s))
	if n > 0 {
		for i := range v {
			v[i] /= n
		}
	}
}

// CosineSimilarity returns the dot product of two vectors.
// For L2-normalized vectors this equals the cosine similarity.
func CosineSimilarity(a, b []float32) float64 {
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return dot
}

// Sigmoid returns the sigmoid activation of x.
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Softmax returns the softmax probabilities for the given logits.
func Softmax(logits []float64) []float64 {
	max := logits[0]
	for _, l := range logits[1:] {
		if l > max {
			max = l
		}
	}
	var sum float64
	probs := make([]float64, len(logits))
	for i, l := range logits {
		probs[i] = math.Exp(l - max)
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}
