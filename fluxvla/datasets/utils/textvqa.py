# Copyright 2026 Limx Dynamics
"""Read explicit official TextVQA splits and answer-free OCR prompts."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TEXTVQA_VAL_URL = (
    'https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json')


def load_textvqa_annotations(annotation_path, split):
    """Read one explicit official split; never reinterpret val as training."""
    if split not in ('train', 'val'):
        raise ValueError('TextVQA split must be train or val.')
    path = Path(annotation_path)
    if not path.is_file():
        raise FileNotFoundError(
            f'TextVQA annotations missing: {path}. '
            'Download the official JSON from '
            f'{TEXTVQA_VAL_URL.replace("_val", "_" + split)} '
            'to this path.')
    payload = json.loads(path.read_text(encoding='utf-8'))
    rows = payload.get('data') if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected official TextVQA JSON with nonempty data.')
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or row.get('set_name') != split
                or not isinstance(row.get('question'), str)
                or not row['question'].strip()
                or type(row.get('question_id')) is not int
                or not isinstance(row.get('image_id'), str)
                or not re.fullmatch(r'[A-Za-z0-9_-]+', row['image_id'])
                or not isinstance(row.get('answers'), list)
                or len(row['answers']) != 10
                or not all(isinstance(a, str) for a in row['answers'])):
            raise ValueError(f'Invalid TextVQA {split} record.')
        if row['question_id'] in seen:
            raise ValueError('Duplicate TextVQA question_id.')
        seen.add(row['question_id'])
    return rows


def load_textvqa_validation(annotation_path):
    return load_textvqa_annotations(annotation_path, 'val')


def validate_textvqa_images(rows, image_root):
    """Check unique images; missing media must never change the eval sample."""
    root = Path(image_root)
    paths = [
        root / f'{image_id}.jpg'
        for image_id in sorted({row['image_id']
                                for row in rows})
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        exists = list(pool.map(Path.is_file, paths))
    missing = [str(path) for path, found in zip(paths, exists) if not found]
    if missing:
        raise FileNotFoundError(
            f'Missing {len(missing)} TextVQA images; first: {missing[0]}')
    return len(paths)


def load_textvqa_ocr_prompts(question_path, rows):
    """Join LLaVA-1.5 OCR questions to official validation question IDs.

    LLaVA's ``question_id`` is an IMAGE ID and is not unique. Join on image
    AND question instead; never drop repeated images or zip files by order.
    Accept only the published three-line prompt schema, not model answers.
    Call with the full validation split, before selecting any smoke subset.
    """
    expected = {(row['image_id'], row['question'].lower()): row['question_id']
                for row in rows}
    if len(expected) != len(rows):
        raise ValueError('Ambiguous TextVQA image/question pairs.')
    prompts = {}
    with Path(question_path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if (not isinstance(item, dict) or
                    set(item) - {'question_id', 'image', 'text', 'category'}
                    or not isinstance(item.get('text'), str)
                    or not isinstance(item.get('image'), str)):
                raise ValueError(
                    f'Invalid OCR question record at line {number}.')
            parts = item['text'].split('\n')
            if (len(parts) != 3
                    or not parts[1].startswith('Reference OCR token: ')
                    or parts[2] !=
                    'Answer the question using a single word or phrase.'):
                raise ValueError(
                    f'Expected LLaVA-1.5 OCR prompt at line {number}.')
            key = (item['image'].removesuffix('.jpg'), parts[0].lower())
            question_id = expected.get(key)
            if (question_id is None or item['image'] != key[0] + '.jpg'
                    or item.get('question_id') not in (key[0], question_id)):
                raise ValueError(
                    f'OCR image/question mismatch at line {number}.')
            if question_id in prompts:
                raise ValueError('Duplicate TextVQA OCR image/question pair.')
            prompts[question_id] = item['text']
    if len(prompts) != len(rows):
        raise ValueError(
            'OCR questions must cover the entire validation split; '
            f'found {len(prompts)} of {len(rows)}.')
    return prompts
