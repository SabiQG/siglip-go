package siglip

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

type tokenizer struct {
	pieceToID   map[string]int
	scores      []float64
	padID       int
	unkID       int
	eosID       int
	maxPieceLen int // in runes
}

func loadSigLIPTokenizer(vocabPath string) (*tokenizer, error) {
	data, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, err
	}
	var raw struct {
		Vocab []struct {
			ID    int     `json:"id"`
			Piece string  `json:"piece"`
			Score float64 `json:"score"`
		} `json:"vocab"`
		PadID int `json:"pad_id"`
		UnkID int `json:"unk_id"`
		EosID int `json:"eos_id"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	pieceToID := make(map[string]int, len(raw.Vocab))
	scores := make([]float64, len(raw.Vocab))
	maxPieceLen := 0
	for _, v := range raw.Vocab {
		pieceToID[v.Piece] = v.ID
		scores[v.ID] = v.Score
		rl := utf8.RuneCountInString(v.Piece)
		if rl > maxPieceLen {
			maxPieceLen = rl
		}
	}

	return &tokenizer{
		pieceToID:   pieceToID,
		scores:      scores,
		padID:       raw.PadID,
		unkID:       raw.UnkID,
		eosID:       raw.EosID,
		maxPieceLen: maxPieceLen,
	}, nil
}

func (t *tokenizer) normalizeText(text string) string {
	text = strings.ToLower(text)
	text = norm.NFKC.String(text)
	text = "▁" + strings.ReplaceAll(text, " ", "▁")
	return text
}

func (t *tokenizer) encode(text string) []int64 {
	text = t.normalizeText(text)
	runes := []rune(text)
	n := len(runes)

	// Viterbi forward pass
	bestScore := make([]float64, n+1)
	bestPrev := make([]int, n+1)
	for i := range bestScore {
		bestScore[i] = math.Inf(-1)
		bestPrev[i] = -1
	}
	bestScore[0] = 0

	for i := 0; i < n; i++ {
		if math.IsInf(bestScore[i], -1) {
			continue
		}
		maxLen := t.maxPieceLen
		if i+maxLen > n {
			maxLen = n - i
		}
		for l := 1; l <= maxLen; l++ {
			piece := string(runes[i : i+l])
			if id, ok := t.pieceToID[piece]; ok {
				score := bestScore[i] + t.scores[id]
				if score > bestScore[i+l] {
					bestScore[i+l] = score
					bestPrev[i+l] = i
				}
			}
		}
		// Byte fallback: ensure we can always advance at least 1 rune
		if math.IsInf(bestScore[i+1], -1) {
			bestScore[i+1] = bestScore[i] - 100.0
			bestPrev[i+1] = i
		}
	}

	// Traceback
	var pieces []string
	pos := n
	for pos > 0 {
		prev := bestPrev[pos]
		pieces = append(pieces, string(runes[prev:pos]))
		pos = prev
	}
	// Reverse
	for i, j := 0, len(pieces)-1; i < j; i, j = i+1, j-1 {
		pieces[i], pieces[j] = pieces[j], pieces[i]
	}

	// Map pieces to token IDs
	var tokenIDs []int64
	for _, p := range pieces {
		if id, ok := t.pieceToID[p]; ok {
			tokenIDs = append(tokenIDs, int64(id))
		} else {
			// Byte fallback: encode character as UTF-8 bytes
			for _, b := range []byte(p) {
				byteID := int(b) + 3 // <0x00> is at id 3
				if byteID < len(t.scores) {
					tokenIDs = append(tokenIDs, int64(byteID))
				} else {
					tokenIDs = append(tokenIDs, int64(t.unkID))
				}
			}
		}
	}

	// Build padded output: tokens + EOS + padding (all with eosID since eos==pad)
	ids := make([]int64, maxTextLen)
	for i := range ids {
		ids[i] = int64(t.eosID) // pad with EOS (==PAD)
	}
	limit := len(tokenIDs)
	if limit > maxTextLen-1 { // leave room for EOS
		limit = maxTextLen - 1
	}
	for i := 0; i < limit; i++ {
		ids[i] = tokenIDs[i]
	}
	// Position limit is EOS, rest already filled with eosID
	ids[limit] = int64(t.eosID)

	return ids
}
