"""Option A: raw English news text -> NER -> structured JSON with character offsets."""
import argparse
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from baseline_ner import ROOT, LABELS, align, spans, counts
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

CHUNK_WORDS=64

def chunks(rows, tokenizer):
    items=[]; mapping=[]
    for ri,row in enumerate(rows):
        for start in range(0,len(row['tokens']),CHUNK_WORDS):
            words=row['tokens'][start:start+CHUNK_WORDS]; tags=row['tags'][start:start+CHUNK_WORDS]
            item=align(tokenizer,words,tags)
            if len(item['input_ids'])>tokenizer.model_max_length:
                raise ValueError('Chunk exceeds model capacity; reduce CHUNK_WORDS. No silent truncation.')
            items.append(dict(item)); mapping.append((ri,start,len(words)))
    return items,mapping

@torch.no_grad()
def predict(model, tokenizer, rows, device):
    items,mapping=chunks(rows,tokenizer)
    loader=DataLoader(items,batch_size=32,collate_fn=DataCollatorForTokenClassification(tokenizer))
    predictions=[['O']*len(r['tokens']) for r in rows]; offset=0
    model.eval()
    for batch in loader:
        labels=batch.pop('labels')
        logits=model(**{k:v.to(device) for k,v in batch.items()}).logits.cpu()
        for lab, pred in zip(labels,logits.argmax(-1)):
            ri,start,n=mapping[offset]; offset+=1
            tags=[LABELS[int(p)] for l,p in zip(lab,pred) if int(l)!=-100]
            assert len(tags)==n
            predictions[ri][start:start+n]=tags
    return predictions


NAMES={'PER':'PERSON','ORG':'ORGANIZATION','LOC':'LOCATION','MISC':'MISC'}

class InformationExtractor:
    def __init__(self, model_path=None):
        self.path=model_path or ROOT/'training_output/ner_model'
        self.tokenizer=AutoTokenizer.from_pretrained(self.path,use_fast=True,model_max_length=512)
        self.model=AutoModelForTokenClassification.from_pretrained(self.path)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); self.model.to(self.device).eval()

    def extract(self,text):
        matches=list(re.finditer(r'\w+|[^\w\s]',text))
        if not matches:
            return {'PERSON':[],'ORGANIZATION':[],'LOCATION':[],'MISC':[],'entities':[]}
        words=[m.group() for m in matches]
        tags=predict(self.model,self.tokenizer,[{'tokens':words,'tags':['O']*len(words)}],self.device)[0]
        result={key:[] for key in NAMES.values()}; result['entities']=[]
        for a,b,t in sorted(spans(tags)):
            start,end=matches[a].start(),matches[b-1].end(); surface=text[start:end]; label=NAMES[t]
            result['entities'].append(dict(text=surface,label=label,start=start,end=end))
            if surface not in result[label]: result[label].append(surface)
        return result

def evaluate_ie():
    gold=json.loads((ROOT/'ie_gold.json').read_text(encoding='utf-8'))
    extractor=InformationExtractor(); gs,ps=set(),set(); records=[]
    for i,row in enumerate(gold):
        output=extractor.extract(row['text']); records.append(dict(**row,prediction=output))
        gs|={(i,e['start'],e['end'],e['label']) for e in row['entities']}
        ps|={(i,e['start'],e['end'],e['label']) for e in output['entities']}
    metrics=dict(sentences=len(gold),**counts(gs,ps))
    (ROOT/'training_output/ie_predictions.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'training_output/ie_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('IE independent authored news-style set:',metrics,flush=True)
    return metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('text',nargs='?',default='Barack Obama visited Paris.')
    parser.add_argument('--model'); parser.add_argument('--evaluate',action='store_true'); args=parser.parse_args()
    if args.evaluate: print(evaluate_ie())
    else: print(json.dumps(InformationExtractor(args.model).extract(args.text),ensure_ascii=False,indent=2))
