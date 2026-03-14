import sentencepiece as spm
import json

sp = spm.SentencePieceProcessor()
sp.Load('siglip_onnx/spiece.model')

print('Vocab size:', sp.GetPieceSize())
print('Pad id:', sp.pad_id())
print('Unk id:', sp.unk_id())
print('Bos id:', sp.bos_id())
print('Eos id:', sp.eos_id())

ids = sp.EncodeAsIds('hello world')
print('hello world ids:', ids)

pieces = sp.EncodeAsPieces('hello world')
print('hello world pieces:', pieces)

ids2 = sp.EncodeAsIds('A dog on the beach')
print('A dog on the beach ids:', ids2)

pieces2 = sp.EncodeAsPieces('A dog on the beach')
print('A dog on the beach pieces:', pieces2)

vocab_data = []
for i in range(sp.GetPieceSize()):
    vocab_data.append({
        'id': i,
        'piece': sp.IdToPiece(i),
        'score': sp.GetScore(i),
    })

with open('siglip_onnx/vocab.json', 'w') as f:
    json.dump({
        'vocab': vocab_data,
        'pad_id': sp.pad_id(),
        'unk_id': sp.unk_id(),
        'eos_id': sp.eos_id(),
        'bos_id': sp.bos_id(),
    }, f)
print('Exported vocab.json with', len(vocab_data), 'entries')
