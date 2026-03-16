package tests

import (
	"math"
	"testing"

	siglip "github.com/sabi/siglip-go"
)

const embDim = 768

func approxF32(a, b float32, tol float64) bool {
	return math.Abs(float64(a-b)) < tol
}

func approxF64(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

// --- Normalize ---

func TestNormalize_UnitVector(t *testing.T) {
	v := []float32{3, 4}
	siglip.Normalize(v)
	if !approxF32(v[0], 0.6, 1e-6) || !approxF32(v[1], 0.8, 1e-6) {
		t.Errorf("Normalize([3,4]) = %v, want [0.6, 0.8]", v)
	}
}

func TestNormalize_AlreadyUnit(t *testing.T) {
	v := []float32{1, 0, 0}
	siglip.Normalize(v)
	if v[0] != 1 || v[1] != 0 || v[2] != 0 {
		t.Errorf("Normalize([1,0,0]) = %v, want [1,0,0]", v)
	}
}

func TestNormalize_ZeroVector(t *testing.T) {
	v := []float32{0, 0, 0}
	siglip.Normalize(v)
	for i, x := range v {
		if x != 0 {
			t.Errorf("Normalize(zero)[%d] = %v, want 0", i, x)
		}
	}
}

func TestNormalize_ResultHasUnitLength(t *testing.T) {
	v := []float32{1, 2, 3, 4, 5}
	siglip.Normalize(v)
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	if !approxF64(math.Sqrt(n), 1.0, 1e-5) {
		t.Errorf("||Normalize(v)|| = %v, want 1.0", math.Sqrt(n))
	}
}

func TestNormalize_Negative(t *testing.T) {
	v := []float32{-3, 4}
	siglip.Normalize(v)
	if !approxF32(v[0], -0.6, 1e-6) || !approxF32(v[1], 0.8, 1e-6) {
		t.Errorf("Normalize([-3,4]) = %v, want [-0.6, 0.8]", v)
	}
}

// --- CosineSimilarity ---

func TestCosineSimilarity_Identical(t *testing.T) {
	a := []float32{0.6, 0.8}
	got := siglip.CosineSimilarity(a, a)
	if !approxF64(got, 1.0, 1e-6) {
		t.Errorf("CosineSimilarity(a, a) = %v, want 1.0", got)
	}
}

func TestCosineSimilarity_Orthogonal(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}
	got := siglip.CosineSimilarity(a, b)
	if !approxF64(got, 0.0, 1e-6) {
		t.Errorf("CosineSimilarity([1,0], [0,1]) = %v, want 0.0", got)
	}
}

func TestCosineSimilarity_Opposite(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{-1, 0}
	got := siglip.CosineSimilarity(a, b)
	if !approxF64(got, -1.0, 1e-6) {
		t.Errorf("CosineSimilarity([1,0], [-1,0]) = %v, want -1.0", got)
	}
}

func TestCosineSimilarity_HighDim(t *testing.T) {
	a := make([]float32, embDim)
	b := make([]float32, embDim)
	for i := range a {
		a[i] = float32(i)
		b[i] = float32(i)
	}
	siglip.Normalize(a)
	siglip.Normalize(b)
	got := siglip.CosineSimilarity(a, b)
	if !approxF64(got, 1.0, 1e-4) {
		t.Errorf("CosineSimilarity(same_768d) = %v, want 1.0", got)
	}
}

// --- Sigmoid ---

func TestSigmoid_Zero(t *testing.T) {
	got := siglip.Sigmoid(0)
	if !approxF64(got, 0.5, 1e-10) {
		t.Errorf("Sigmoid(0) = %v, want 0.5", got)
	}
}

func TestSigmoid_LargePositive(t *testing.T) {
	got := siglip.Sigmoid(100)
	if !approxF64(got, 1.0, 1e-10) {
		t.Errorf("Sigmoid(100) = %v, want ~1.0", got)
	}
}

func TestSigmoid_LargeNegative(t *testing.T) {
	got := siglip.Sigmoid(-100)
	if !approxF64(got, 0.0, 1e-10) {
		t.Errorf("Sigmoid(-100) = %v, want ~0.0", got)
	}
}

func TestSigmoid_KnownValue(t *testing.T) {
	got := siglip.Sigmoid(1)
	want := 1.0 / (1.0 + math.Exp(-1))
	if !approxF64(got, want, 1e-10) {
		t.Errorf("Sigmoid(1) = %v, want %v", got, want)
	}
}

func TestSigmoid_Symmetry(t *testing.T) {
	for _, x := range []float64{0.5, 1.0, 2.5, 10.0} {
		sum := siglip.Sigmoid(x) + siglip.Sigmoid(-x)
		if !approxF64(sum, 1.0, 1e-10) {
			t.Errorf("Sigmoid(%v) + Sigmoid(%v) = %v, want 1.0", x, -x, sum)
		}
	}
}

// --- Softmax ---

func TestSoftmax_SumsToOne(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0}
	probs := siglip.Softmax(logits)
	var sum float64
	for _, p := range probs {
		sum += p
	}
	if !approxF64(sum, 1.0, 1e-10) {
		t.Errorf("Softmax sum = %v, want 1.0", sum)
	}
}

func TestSoftmax_Ordering(t *testing.T) {
	probs := siglip.Softmax([]float64{1.0, 3.0, 2.0})
	if probs[1] <= probs[2] || probs[2] <= probs[0] {
		t.Errorf("Softmax ordering wrong: %v", probs)
	}
}

func TestSoftmax_Uniform(t *testing.T) {
	probs := siglip.Softmax([]float64{5.0, 5.0, 5.0})
	for i, p := range probs {
		if !approxF64(p, 1.0/3.0, 1e-10) {
			t.Errorf("Softmax(uniform)[%d] = %v, want 1/3", i, p)
		}
	}
}

func TestSoftmax_LargeValues(t *testing.T) {
	probs := siglip.Softmax([]float64{1000, 1001, 1002})
	var sum float64
	for _, p := range probs {
		sum += p
		if math.IsNaN(p) || math.IsInf(p, 0) {
			t.Fatalf("Softmax produced NaN/Inf: %v", probs)
		}
	}
	if !approxF64(sum, 1.0, 1e-10) {
		t.Errorf("Softmax(large) sum = %v, want 1.0", sum)
	}
}

// --- benchmarks ---

func BenchmarkNormalize768(b *testing.B) {
	v := make([]float32, embDim)
	for i := range v {
		v[i] = float32(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		siglip.Normalize(v)
	}
}

func BenchmarkCosineSimilarity768(b *testing.B) {
	a := make([]float32, embDim)
	bv := make([]float32, embDim)
	for i := range a {
		a[i] = float32(i)
		bv[i] = float32(embDim - i)
	}
	siglip.Normalize(a)
	siglip.Normalize(bv)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		siglip.CosineSimilarity(a, bv)
	}
}

func BenchmarkSigmoid(b *testing.B) {
	for i := 0; i < b.N; i++ {
		siglip.Sigmoid(float64(i % 20))
	}
}
