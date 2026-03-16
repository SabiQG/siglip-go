package tests

import (
	"testing"
)

// Tokenizer internals are unexported, so we test tokenization behavior
// indirectly through the public Classify API. Correct classification
// proves the tokenizer works: different labels → different token IDs →
// different text embeddings → different scores.

func TestTokenizer_DifferentLabelsProduceDifferentScores(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	results, err := c.Classify("images/perro_playa.jpg", []string{"a dog", "a cat"})
	if err != nil {
		t.Fatalf("Classify: %v", err)
	}
	if results[0].Score == results[1].Score {
		t.Error("different labels should produce different scores (proves tokenizer differentiates)")
	}
}

func TestTokenizer_Multilingual(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	labels := []string{
		"un perro",   // Spanish
		"eine Katze", // German
		"猫",          // Chinese
		"самолёт",    // Russian
	}
	results, err := c.Classify("images/perro_playa.jpg", labels)
	if err != nil {
		t.Fatalf("Classify with multilingual labels: %v", err)
	}
	if len(results) != len(labels) {
		t.Errorf("got %d results, want %d", len(results), len(labels))
	}
	// "un perro" (Spanish for "a dog") should rank highest for a dog image
	if results[0].Label != "un perro" {
		t.Errorf("top label = %q, want 'un perro'", results[0].Label)
	}
}

func TestTokenizer_LongText(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	long := "this is a very long sentence that should produce many tokens and eventually get truncated to the maximum sequence length allowed by the model configuration"
	_, err := c.Classify("images/f35.jpg", []string{long, "jet"})
	if err != nil {
		t.Fatalf("Classify with long label: %v", err)
	}
}

func TestTokenizer_EmptyLabel(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	_, err := c.Classify("images/f35.jpg", []string{"", "jet"})
	if err != nil {
		t.Fatalf("Classify with empty label should not crash: %v", err)
	}
}

func TestTokenizer_Deterministic(t *testing.T) {
	c := newTestClassifier(t)
	defer c.Close()

	labels := []string{"a photo of a dog", "a photo of a cat"}
	r1, err := c.Classify("images/perro_playa.jpg", labels)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := c.Classify("images/perro_playa.jpg", labels)
	if err != nil {
		t.Fatal(err)
	}
	for i := range r1 {
		if r1[i].Score != r2[i].Score {
			t.Errorf("Classify not deterministic: result[%d].Score=%f vs %f", i, r1[i].Score, r2[i].Score)
		}
	}
}
