package tests

import (
	"testing"

	siglip "github.com/sabi/siglip-go"
)

const (
	dogImage        = "images/dog_beach.jpg"
	f35Image        = "images/f35.jpg"
	fighterJetImage = "images/fighter_jet.jpg"
)

func newTestClassifier(t *testing.T) *siglip.Classifier {
	t.Helper()

	c, err := siglip.New()
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	return c
}

// --- Classify ---

func TestClassify_ReturnsAllLabels(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	labels := []string{"a dog", "a cat", "a plane"}
	results, err := c.Classify(dogImage, labels)
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if len(results) != len(labels) {
		t.Errorf("got %d results, want %d", len(results), len(labels))
	}
	for _, r := range results {
		t.Logf("  %-30s  score=%.4f  cosine=%.4f", r.Label, r.Score, r.Cosine)
	}
}

func TestClassify_SortedDescending(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	results, err := c.Classify(dogImage, []string{"a dog", "a cat", "a car"})
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: [%d].Score=%f > [%d].Score=%f",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}
	for _, r := range results {
		t.Logf("  %-30s  score=%.4f  cosine=%.4f", r.Label, r.Score, r.Cosine)
	}
}

func TestClassify_ScoresInRange(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	results, err := c.Classify(f35Image, []string{"jet", "flower"})
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	for _, r := range results {
		if r.Score < 0 || r.Score > 1 {
			t.Errorf("score out of [0,1]: %f for %q", r.Score, r.Label)
		}
		if r.Cosine < -1 || r.Cosine > 1 {
			t.Errorf("cosine out of [-1,1]: %f for %q", r.Cosine, r.Label)
		}
	}
}

func TestClassify_CorrectLabel(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	results, err := c.Classify(dogImage, []string{"a dog on the beach", "a cat indoors", "an airplane"})
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if results[0].Label != "a dog on the beach" {
		t.Errorf("top label = %q, want 'a dog on the beach'", results[0].Label)
	}
	for _, r := range results {
		t.Logf("  %-30s  score=%.4f  cosine=%.4f", r.Label, r.Score, r.Cosine)
	}
}

func TestClassify_InvalidImage(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	_, err := c.Classify("/nonexistent/img.jpg", []string{"a"})
	if err == nil {
		t.Error("Classify with invalid image should return error")
	}
}

// --- Search ---

func TestSearch_TopK(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	imgs := []string{dogImage, f35Image, fighterJetImage}
	embs := make([][]float32, len(imgs))
	for i, img := range imgs {
		var err error
		embs[i], err = c.ImageEmbedding(img)
		if err != nil {
			t.Fatalf("ImageEmbedding(%s): %v", img, err)
		}
	}

	matches, err := c.Search(embs, "a dog", 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(matches) != 2 {
		t.Errorf("Search returned %d results, want 2", len(matches))
	}
	for _, m := range matches {
		t.Logf("  [%d] %-20s  score=%.4f  cosine=%.4f", m.Index, imgs[m.Index], m.Score, m.Cosine)
	}
}

func TestSearch_SortedDescending(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	imgs := []string{dogImage, f35Image, fighterJetImage}
	embs := make([][]float32, len(imgs))
	for i, img := range imgs {
		embs[i], _ = c.ImageEmbedding(img)
	}

	matches, err := c.Search(embs, "aircraft", 3)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for i := 1; i < len(matches); i++ {
		if matches[i].Score > matches[i-1].Score {
			t.Errorf("Search results not sorted: [%d].Score=%f > [%d].Score=%f",
				i, matches[i].Score, i-1, matches[i-1].Score)
		}
	}
}

func TestSearch_KLargerThanEmbeddings(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	emb, _ := c.ImageEmbedding(dogImage)
	matches, err := c.Search([][]float32{emb}, "dog", 100)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(matches) != 1 {
		t.Errorf("Search with k>len should clamp: got %d, want 1", len(matches))
	}
}

func TestSearch_IndexPreserved(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	imgs := []string{dogImage, f35Image}
	embs := make([][]float32, len(imgs))
	for i, img := range imgs {
		embs[i], _ = c.ImageEmbedding(img)
	}

	matches, err := c.Search(embs, "a dog on the beach", 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if matches[0].Index != 0 {
		t.Errorf("Search top match index = %d, want 0 (dog image)", matches[0].Index)
	}
	for _, m := range matches {
		t.Logf("  [%d] %-20s  score=%.4f  cosine=%.4f", m.Index, imgs[m.Index], m.Score, m.Cosine)
	}
}

// --- Options ---

func TestWithNoDownload_MissingModels(t *testing.T) {
	_, err := siglip.New(siglip.WithNoDownload(), siglip.WithModelDir("/tmp/nonexistent_siglip_models"))
	if err == nil {
		t.Error("New with WithNoDownload and missing models should return error")
	}
}

// --- benchmarks ---

func BenchmarkClassify(b *testing.B) {
	c, err := siglip.New()
	if err != nil {
		b.Fatalf("New(): %v", err)
	}
	defer c.Close()

	labels := []string{"a dog", "a cat", "an airplane"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Classify(dogImage, labels)
	}
}

func BenchmarkImageEmbedding(b *testing.B) {
	c, err := siglip.New()
	if err != nil {
		b.Fatalf("New(): %v", err)
	}
	defer c.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.ImageEmbedding(dogImage)
	}
}
