"""Shared CoNLL-2003 loading, strict BIO metrics and word/subword alignment."""
from pathlib import Path
from collections import Counter
import json
import hashlib
import urllib.request
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
LABELS = ['O','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC','B-MISC','I-MISC']
TYPES = ['PER','ORG','LOC','MISC']
MODEL = 'google/bert_uncased_L-2_H-128_A-2'
SEED = 42

def load_data():
    folder = ROOT / 'datasets' / 'conll2003'
    folder.mkdir(parents=True, exist_ok=True)
    result = {}
    manifest = {}
    for split in ['train','validation','test']:
        path = folder / f'{split}.parquet'
        if not path.exists():
            url = f'https://huggingface.co/datasets/eriktks/conll2003/resolve/refs%2Fconvert%2Fparquet/conll2003/{split}/0000.parquet'
            urllib.request.urlretrieve(url, path)
        frame = pd.read_parquet(path)
        manifest[split] = dict(path=str(path.relative_to(ROOT)), sentences=len(frame),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        result[split] = [{'id': str(r['id']), 'tokens': list(r['tokens']),
                          'tags': [LABELS[int(i)] for i in r['ner_tags']]} for r in frame.to_dict('records')]
    output = ROOT / 'training_output'
    output.mkdir(exist_ok=True)
    (output / 'data_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return result

def spans(tags):
    """Strict IOB2: illegal I-X cannot open an entity (counted as invalid separately)."""
    out, start, kind = set(), None, None
    for i, tag in enumerate(list(tags) + ['O']):
        prefix, _, typ = tag.partition('-')
        continues = prefix == 'I' and typ == kind and start is not None
        if start is not None and not continues:
            out.add((start, i, kind)); start, kind = None, None
        if prefix == 'B':
            start, kind = i, typ
    return out

def counts(gold, pred):
    tp = len(gold & pred); fp = len(pred - gold); fn = len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.
    r = tp / (tp + fn) if tp + fn else 0.
    return dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0., tp=tp, fp=fp, fn=fn, support=tp+fn)

def evaluate(rows, predictions):
    assert len(rows) == len(predictions)
    gold, pred = set(), set()
    for i, (row, tags) in enumerate(zip(rows, predictions)):
        assert len(row['tokens']) == len(tags)
        gold |= {(i, *s) for s in spans(row['tags'])}
        pred |= {(i, *s) for s in spans(tags)}
    overall = counts(gold, pred)
    y = [t for row in rows for t in row['tags']]; p = [t for ts in predictions for t in ts]
    pr, re, f1, _ = precision_recall_fscore_support(y, p, labels=LABELS[1:], average='micro', zero_division=0)
    overall.update(token_accuracy=accuracy_score(y,p), token_precision=pr, token_recall=re, token_f1=f1,
                   invalid_i_tokens=sum(t.startswith('I-') and (j == 0 or ts[j-1] not in ['B-'+t[2:],t]) for ts in predictions for j,t in enumerate(ts)))
    by_type = [dict(entity=t, **counts({s for s in gold if s[-1]==t}, {s for s in pred if s[-1]==t})) for t in TYPES]
    bucket = lambda s: str(s[2]-s[1]) if s[2]-s[1]<4 else '4+'
    by_length = [dict(length=b, **counts({s for s in gold if bucket(s)==b}, {s for s in pred if bucket(s)==b})) for b in ['1','2','3','4+']]
    return overall, by_type, by_length

def align(tokenizer, words, tags):
    encoding = tokenizer(words, is_split_into_words=True, truncation=False)
    previous = None; aligned = []
    for wid in encoding.word_ids():
        aligned.append(-100 if wid is None or wid == previous else LABELS.index(tags[wid]))
        previous = wid
    encoding['labels'] = aligned
    return encoding

def save_predictions(name, rows, predictions):
    out = ROOT / 'training_output'; out.mkdir(exist_ok=True)
    (out/f'{name}_predictions.json').write_text(json.dumps([dict(**r,prediction=p) for r,p in zip(rows,predictions)],ensure_ascii=False,indent=2),encoding='utf-8')


"""LAB 09 baseline: sparse local token features + Logistic Regression."""
import argparse
import pickle
import time
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def sparse32(x):
    x=csr_matrix(x)
    x.indices=x.indices.astype(np.int32)
    x.indptr=x.indptr.astype(np.int32)
    return x

def features(words, i):
    w = words[i]
    return {'word': w.lower(), 'title': w.istitle(), 'upper': w.isupper(), 'digit': w.isdigit(),
            'prefix2': w[:2].lower(), 'prefix3': w[:3].lower(), 'suffix2': w[-2:].lower(), 'suffix3': w[-3:].lower(),
            'previous': words[i-1].lower() if i else '<BOS>', 'next': words[i+1].lower() if i+1<len(words) else '<EOS>',
            'previous_title': i>0 and words[i-1].istitle(), 'next_title': i+1<len(words) and words[i+1].istitle()}

def predict(model, rows):
    vectorizer, classifier = model
    flat = classifier.predict(sparse32(vectorizer.transform(features(r['tokens'],i) for r in rows for i in range(len(r['tokens'])))))
    result=[]; offset=0
    for row in rows:
        result.append(flat[offset:offset+len(row['tokens'])].tolist()); offset += len(row['tokens'])
    return result

def train(rows):
    vectorizer=DictVectorizer()
    x=vectorizer.fit_transform(features(r['tokens'],i) for r in rows for i in range(len(r['tokens'])))
    y=[t for r in rows for t in r['tags']]
    classifier=LogisticRegression(max_iter=200, solver='saga', random_state=SEED, tol=0.002)
    classifier.fit(sparse32(x),y)
    return vectorizer,classifier

def run(data=None):
    data=data or load_data(); start=time.perf_counter(); model=train(data['train'])
    seconds=time.perf_counter()-start
    output=ROOT/'training_output'; output.mkdir(exist_ok=True)
    with (output/'baseline.pkl').open('wb') as f: pickle.dump(model,f)
    predictions=predict(model,data['test']); save_predictions('baseline',data['test'],predictions)
    metrics, types, lengths=evaluate(data['test'],predictions)
    return dict(model='baseline',train_seconds=seconds,**metrics), types, lengths, predictions

if __name__=='__main__':
    print(run()[0])
