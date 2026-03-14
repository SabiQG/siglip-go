import sentencepiece as spm
import json

sp = spm.SentencePieceProcessor()
sp.Load('siglip_onnx/spiece.model')

# Check byte fallback tokens
byte_tokens = []
for i in range(sp.GetPieceSize()):
    p = sp.IdToPiece(i)
    if p.startswith('<0x') and p.endswith('>'):
        byte_tokens.append((i, p))
print(f"Byte fallback tokens: {len(byte_tokens)}")
if byte_tokens:
    print(f"  First: {byte_tokens[0]}, Last: {byte_tokens[-1]}")

# Show first 10 vocab entries
for i in range(10):
    print(f"  {i}: '{sp.IdToPiece(i)}' score={sp.GetScore(i)}")

# Test the exact encoding SigLIP HF tokenizer produces
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('google/siglip-base-patch16-256-multilingual')

tests = [
    'hello world',
    'a dog on the beach',
    'fighter jet',
    'avión de combate',
    'un perro marrón en la playa',
]
for t in tests:
    enc = tok(t, padding='max_length', max_length=64, truncation=True)
    ids = enc['input_ids']
    non_pad = [x for x in ids if x != 1]
    print(f"\n'{t}' -> {non_pad}")
    sp_ids = sp.EncodeAsIds(t.lower())
    print(f"  SP direct (lowered): {sp_ids}")
    sp_pieces = sp.EncodeAsPieces(t.lower())
    print(f"  SP pieces: {sp_pieces}")
