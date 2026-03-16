package siglip_test

import (
	"fmt"
	"log"

	siglip "github.com/SabiQG/siglip-go"
)

func Example() {
	c, err := siglip.New()
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	results, err := c.Classify("photo.jpg", []string{"a dog", "a cat", "a car"})
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range results {
		fmt.Printf("%-20s  %.1f%%\n", r.Label, r.Score*100)
	}
}

func ExampleClassifier_Classify() {
	c, err := siglip.New()
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	results, err := c.Classify("photo.jpg", []string{
		"a photo of food",
		"a photo of a person",
		"a landscape photo",
	})
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range results {
		fmt.Printf("%-30s  score=%.3f  cosine=%.3f\n", r.Label, r.Score, r.Cosine)
	}
}

func ExampleClassifier_Search() {
	c, err := siglip.New()
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	images := []string{"beach.jpg", "city.jpg", "forest.jpg"}
	embs := make([][]float32, len(images))
	for i, img := range images {
		embs[i], err = c.ImageEmbedding(img)
		if err != nil {
			log.Fatal(err)
		}
	}

	matches, err := c.Search(embs, "a sunset", 2)
	if err != nil {
		log.Fatal(err)
	}
	for _, m := range matches {
		fmt.Printf("#%d %s  score=%.3f\n", m.Index, images[m.Index], m.Score)
	}
}

func ExampleClassifier_ImageEmbedding() {
	c, err := siglip.New()
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	emb, err := c.ImageEmbedding("photo.jpg")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("embedding dimensions: %d\n", len(emb))
}
