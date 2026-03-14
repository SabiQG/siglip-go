// Command classify demonstrates zero-shot image classification with SigLIP.
package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	siglip "github.com/sabi/siglip-go"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: classify <image> [label1,label2,...]\n")
		os.Exit(1)
	}

	imagePath := os.Args[1]

	labels := []string{
		"a dog on the beach",
		"a cat sitting on a sofa",
		"a car on the road",
		"a person walking in the city",
		"a bird flying in the sky",
		"fighter jet",
		"military aircraft",
	}
	if len(os.Args) > 2 {
		labels = strings.Split(os.Args[2], ",")
	}

	// Create a classifier. Models + ONNX Runtime are auto-downloaded on first use.
	// Override with:
	//   siglip.WithModelDir("/path/to/models")            — custom path
	//   siglip.WithRuntimeLib("/path/to/libonnxruntime")   — custom ORT path
	//   siglip.WithNoDownload()                            — disable auto-download
	c, err := siglip.New()
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	results, err := c.Classify(imagePath, labels)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("🔍 Clasificando imagen: %s\n\n", imagePath)
	fmt.Printf("  %-40s  %8s  %8s\n", "Label", "Cosine", "Score")
	fmt.Printf("  %-40s  %8s  %8s\n",
		strings.Repeat("─", 40), "──────", "─────")
	for _, r := range results {
		fmt.Printf("  %-40s  %8.4f  %6.1f%%\n", r.Label, r.Cosine, r.Score*100)
	}
	fmt.Printf("\n🏆 Best match: \"%s\" (%.1f%%)\n", results[0].Label, results[0].Score*100)
}
