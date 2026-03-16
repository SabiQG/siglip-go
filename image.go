package siglip

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

func preprocessImage(imagePath string) ([]float32, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("error abriendo imagen: %v", err)
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, fmt.Errorf("error decodificando imagen: %v", err)
	}

	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	var nw, nh int
	if w < h {
		nw, nh = imgSize, int(float64(h)*float64(imgSize)/float64(w))
	} else {
		nw, nh = int(float64(w)*float64(imgSize)/float64(h)), imgSize
	}
	resized := image.NewRGBA(image.Rect(0, 0, nw, nh))
	draw.CatmullRom.Scale(resized, resized.Bounds(), img, bounds, draw.Src, nil)

	x0 := (nw - imgSize) / 2
	y0 := (nh - imgSize) / 2
	cropped := image.NewRGBA(image.Rect(0, 0, imgSize, imgSize))
	draw.Draw(cropped, cropped.Bounds(), resized, image.Pt(x0, y0), draw.Src)

	tensor := make([]float32, 3*imgSize*imgSize)
	ch := imgSize * imgSize
	for y := 0; y < imgSize; y++ {
		for x := 0; x < imgSize; x++ {
			c := cropped.RGBAAt(x, y)
			idx := y*imgSize + x
			tensor[idx] = (float32(c.R)/255.0 - meanR) / stdR
			tensor[ch+idx] = (float32(c.G)/255.0 - meanG) / stdG
			tensor[2*ch+idx] = (float32(c.B)/255.0 - meanB) / stdB
		}
	}
	return tensor, nil
}

func (c *Classifier) imageEmbedding(imagePath string) ([]float32, error) {
	input, err := preprocessImage(imagePath)
	if err != nil {
		return nil, err
	}
	inT, err := ort.NewTensor([]int64{1, 3, imgSize, imgSize}, input)
	if err != nil {
		return nil, err
	}
	defer inT.Destroy()

	out := make([]float32, embDim)
	outT, err := ort.NewTensor([]int64{1, embDim}, out)
	if err != nil {
		return nil, err
	}
	defer outT.Destroy()

	modelPath := filepath.Join(c.modelDir, "vision_model.onnx")
	sess, err := ort.NewAdvancedSession(modelPath,
		[]string{"pixel_values"}, []string{"image_embeds"},
		[]ort.Value{inT}, []ort.Value{outT}, nil)
	if err != nil {
		return nil, err
	}
	defer sess.Destroy()

	if err := sess.Run(); err != nil {
		return nil, err
	}
	Normalize(out)
	return out, nil
}
