import json
import os
import subprocess
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data[obj['key']] = obj['caption']
    return data


def align_data(ref_path, cand_path):
    ref_data = load_jsonl(ref_path)
    cand_data = load_jsonl(cand_path)
    common_keys = sorted(set(ref_data.keys()) & set(cand_data.keys()))
    references = [ref_data[k] for k in common_keys]
    candidates = [cand_data[k] for k in common_keys]
    return common_keys, references, candidates


def compute_bleu_scores(references, candidates):
    smooth = SmoothingFunction().method1
    bleu1, bleu2, bleu3, bleu4 = [], [], [], []
    for ref, cand in zip(references, candidates):
        ref_tokens = ref.split()
        cand_tokens = cand.split()
        bleu1.append(sentence_bleu([ref_tokens], cand_tokens, weights=(1,0,0,0), smoothing_function=smooth))
        bleu2.append(sentence_bleu([ref_tokens], cand_tokens, weights=(0.5,0.5,0,0), smoothing_function=smooth))
        bleu3.append(sentence_bleu([ref_tokens], cand_tokens, weights=(0.33,0.33,0.33,0), smoothing_function=smooth))
        bleu4.append(sentence_bleu([ref_tokens], cand_tokens, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth))
    return {
        'BLEU-1': sum(bleu1)/len(bleu1),
        'BLEU-2': sum(bleu2)/len(bleu2),
        'BLEU-3': sum(bleu3)/len(bleu3),
        'BLEU-4': sum(bleu4)/len(bleu4),
    }


def to_coco_format(keys, refs, cands):
    gts = [{'image_id': k, 'caption': r} for k, r in zip(keys, refs)]
    res = [{'image_id': k, 'caption': c} for k, c in zip(keys, cands)]
    with open('refs.json', 'w', encoding='utf-8') as f:
        json.dump(gts, f)
    with open('cands.json', 'w', encoding='utf-8') as f:
        json.dump(res, f)

def compute_spice(spice_jar_path='spice-1.0.jar'):
    cmd = [
        'java', '-jar', spice_jar_path,
        'cands.json', 'refs.json',
        '-cache', './tmp/'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    for line in result.stdout.split('\n'):
        if line.strip().startswith('SPICE'):
            print(line.strip())
            break

if __name__ == '__main__':
    ref_path = 'file1.jsonl'
    cand_path = 'file2.jsonl'
    spice_jar_path = 'spice-1.0.jar' 

    keys, references, candidates = align_data(ref_path, cand_path)

    bleu_scores = compute_bleu_scores(references, candidates)
    print("BLEU分数:")
    for k, v in bleu_scores.items():
        print(f"{k}: {v:.4f}")

    to_coco_format(keys, references, candidates)

    compute_spice(spice_jar_path)
