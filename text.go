package siglip

import (
	"fmt"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
)

func (c *Classifier) textEmbedding(text string) ([]float32, error) {
	ids := c.tok.encode(text)

	idT, err := ort.NewTensor([]int64{1, maxTextLen}, ids)
	if err != nil {
		return nil, err
	}
	defer idT.Destroy()

	out := make([]float32, embDim)
	outT, err := ort.NewTensor([]int64{1, embDim}, out)
	if err != nil {
		return nil, err
	}
	defer outT.Destroy()

	modelPath := filepath.Join(c.modelDir, "text_model.onnx")
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("siglip: creating session options: %w", err)
	}
	defer opts.Destroy()
	if err := opts.SetCpuMemArena(false); err != nil {
		return nil, fmt.Errorf("siglip: disabling CPU mem arena: %w", err)
	}
	if err := opts.SetMemPattern(false); err != nil {
		return nil, fmt.Errorf("siglip: disabling mem pattern: %w", err)
	}

	sess, err := ort.NewAdvancedSession(modelPath,
		[]string{"input_ids"}, []string{"text_embeds"},
		[]ort.Value{idT}, []ort.Value{outT}, opts)
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
