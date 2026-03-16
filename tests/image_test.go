package tests

import (
	"math"
	"os"
	"testing"
)

// --- ImageEmbedding (public API) ---

func TestImageEmbedding_Dimension(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	emb, err := c.ImageEmbedding("images/perro_playa.jpg")
	if err != nil {
		t.Fatalf("ImageEmbedding: %v", err)
	}
	if len(emb) != embDim {
		t.Errorf("embedding dim = %d, want %d", len(emb), embDim)
	}
}

func TestImageEmbedding_Normalized(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	emb, err := c.ImageEmbedding("images/perro_playa.jpg")
	if err != nil {
		t.Fatalf("ImageEmbedding: %v", err)
	}
	var norm float64
	for _, x := range emb {
		norm += float64(x) * float64(x)
	}
	norm = math.Sqrt(norm)
	if !approxF64(norm, 1.0, 1e-4) {
		t.Errorf("embedding L2 norm = %f, want 1.0", norm)
	}
}

func TestImageEmbedding_Deterministic(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	e1, _ := c.ImageEmbedding("images/perro_playa.jpg")
	e2, _ := c.ImageEmbedding("images/perro_playa.jpg")
	for i := range e1 {
		if e1[i] != e2[i] {
			t.Errorf("ImageEmbedding not deterministic at [%d]: %f vs %f", i, e1[i], e2[i])
			break
		}
	}
}

func TestImageEmbedding_DifferentImages(t *testing.T) {
	paths := []string{"images/perro_playa.jpg", "images/f35.jpg"}
	for _, p := range paths {
		if _, err := os.Stat(p); err != nil {
			t.Skipf("test image not found: %s", p)
		}
	}
	c := newTestClassifier(t)
	defer c.Close()

	e1, _ := c.ImageEmbedding(paths[0])
	e2, _ := c.ImageEmbedding(paths[1])
	same := true
	for i := range e1 {
		if e1[i] != e2[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different images should produce different embeddings")
	}
}

func TestImageEmbedding_InvalidPath(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	_, err := c.ImageEmbedding("/nonexistent/image.jpg")
	if err == nil {
		t.Error("ImageEmbedding with invalid path should return error")
	}
}

func TestImageEmbedding_InvalidFormat(t *testing.T) {
	f, err := os.CreateTemp("", "bad_image_*.jpg")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())
	f.WriteString("not an image")
	f.Close()

	c := newTestClassifier(t)
	defer c.Close()

	_, err = c.ImageEmbedding(f.Name())
	if err == nil {
		t.Error("ImageEmbedding with invalid image data should return error")
	}
}
