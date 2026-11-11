"""LAB 08: small BERT, frozen encoder and random initialization on IMDb."""
from pathlib import Path
import argparse
import json
import platform
import random
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL = 'google/bert_uncased_L-2_H-128_A-2'
SEED = 42
ROOT = Path(__file__).resolve().parent


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def select_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: choose cpu or install CUDA-enabled PyTorch.')
    return torch.device(name)


def prepare(path, output):
    """Recover LAB 07's 2800/400/800 split, including duplicate replacement."""
    raw = pd.read_csv(path)
    for label in ['positive', 'negative']:
        if raw.sentiment.eq(label).sum() < 2000:
            raise ValueError('Need at least 2000 examples per class, as in LAB 07.')
    frame = pd.concat([raw[raw.sentiment.eq('positive')].head(2000),
                       raw[raw.sentiment.eq('negative')].head(2000)])
    frame = frame.sample(frac=1, random_state=SEED).reset_index().rename(columns={'index': 'source_row'})
    frame['split'] = ['train'] * 2800 + ['validation'] * 400 + ['test'] * 800
    heldout, used = set(frame.iloc[2800:].review), set(frame.review)
    replacements = []
    for i in range(2800):
        if frame.at[i, 'review'] in heldout:
            candidates = raw[raw.sentiment.eq(frame.at[i, 'sentiment']) & ~raw.review.isin(used)]
            if candidates.empty:
                raise ValueError('No unused review available to remove train/heldout overlap.')
            candidate = candidates.iloc[0]
            replacements.append({'position': i, 'old_row': int(frame.at[i, 'source_row']), 'new_row': int(candidate.name)})
            frame.at[i, 'source_row'] = int(candidate.name)
            frame.at[i, 'review'] = candidate.review
            used.add(candidate.review)
    assert not set(frame.iloc[:2800].review) & heldout
    assert not set(frame.iloc[2800:3200].review) & set(frame.iloc[3200:].review)
    frame[['source_row', 'sentiment', 'split']].to_csv(output / 'split_manifest.csv', index=False)
    (output / 'data_audit.json').write_text(json.dumps(replacements, indent=2), encoding='utf-8')
    return frame


class Classifier(nn.Module):
    def __init__(self, model_name=MODEL, mode='finetuned', config=None):
        super().__init__()
        if mode not in {'scratch', 'frozen', 'finetuned'}:
            raise ValueError(f'Unknown mode: {mode}')
        self.mode = mode
        if mode == 'scratch':
            self.encoder = AutoModel.from_config(config or AutoConfig.from_pretrained(model_name))
        else:
            self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.encoder.config.hidden_size, 2)
        if mode == 'frozen':
            self.encoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.mode == 'frozen':
            self.encoder.eval()  # fixed features: no encoder dropout during head training
        return self

    def forward(self, input_ids, attention_mask):
        representations = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.dropout(representations[:, 0]))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    targets, predictions, probabilities = [], [], []
    loss_sum = 0.0
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        logits = model(ids, mask)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(-1).cpu().tolist())
        probabilities.extend(logits.softmax(-1).cpu().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='macro', zero_division=0)
    return dict(loss=loss_sum / len(targets), accuracy=accuracy_score(targets, predictions),
                precision=precision, recall=recall, f1=f1), predictions, np.asarray(probabilities)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def run(data_path=ROOT / 'datasets/imdb/IMDB Dataset.csv', output_dir=ROOT / 'training_output',
        epochs=4, device='auto', model_name=MODEL, modes=('scratch', 'frozen', 'finetuned'),
        max_length=64, batch_size=32, train_fraction=1.0, export_dir=ROOT):
    if epochs < 1 or batch_size < 1 or max_length < 2:
        raise ValueError('Invalid epochs, batch size or sequence length.')
    if not modes or len(set(modes)) != len(modes) or set(modes) - {'scratch', 'frozen', 'finetuned'}:
        raise ValueError('Choose distinct modes: scratch, frozen, finetuned.')
    if not 0 < train_fraction <= 1:
        raise ValueError('train_fraction must be in (0, 1].')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(device)
    torch.set_num_threads(4)
    seed()
    frame = prepare(data_path, output)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    if max_length > config.max_position_embeddings:
        raise ValueError('max_length exceeds the pretrained positional embedding capacity.')
    tokenizer.save_pretrained(output / 'tokenizer')
    encoded = tokenizer(frame.review.tolist(), padding='max_length', truncation=True,
                        max_length=max_length, return_tensors='pt')
    labels = torch.tensor(frame.sentiment.eq('positive').astype(int).to_numpy())
    # Fixed permutation gives nested training subsets. Held-out indices never change.
    train_indices = np.sort(np.random.default_rng(SEED).permutation(2800)[:round(2800 * train_fraction)])
    train_count = len(train_indices)
    if train_count == 0:
        raise ValueError('Training subset is empty.')
    frame.iloc[train_indices][['source_row', 'sentiment']].to_csv(output / 'train_subset.csv', index=False)
    settings = dict(model=model_name, revision=getattr(config, '_commit_hash', None),
                    seed=SEED, epochs=epochs, batch_size=batch_size, max_length=max_length,
                    optimizer='AdamW', weight_decay=0.01, learning_rates={'scratch': 1e-3, 'frozen': 1e-3, 'finetuned': 2e-5},
                    train=train_count, train_fraction=train_fraction, validation=400, test=800, device=str(device),
                    hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                    torch=torch.__version__, selection='LAB07 first 2000/class, shuffled seed 42, duplicate repair',
                    checkpoint='minimum validation loss', metrics='macro precision/recall/F1', modes=list(modes))
    (output / 'config.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    rows, histories = [], {}
    for mode in modes:
        seed()
        generator = torch.Generator().manual_seed(SEED)
        loaders = [DataLoader(TensorDataset(encoded['input_ids'][a:b], encoded['attention_mask'][a:b], labels[a:b]),
                              batch_size=batch_size, shuffle=(a == 0), generator=generator if a == 0 else None)
                   for a, b in [(0, 2800), (2800, 3200), (3200, 4000)]]
        loaders[0] = DataLoader(TensorDataset(encoded['input_ids'][train_indices],
                                            encoded['attention_mask'][train_indices], labels[train_indices]),
                                batch_size=batch_size, shuffle=True, generator=generator)
        model = Classifier(model_name, mode).to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=settings['learning_rates'][mode], weight_decay=0.01)
        best, best_state, history = float('inf'), None, []
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            total = 0.0
            synchronize(device)
            train_started = time.perf_counter()
            for ids, mask, y in loaders[0]:
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(ids, mask), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * len(y)
            synchronize(device)
            train_seconds = time.perf_counter() - train_started
            metrics, _, _ = evaluate(model, loaders[1], device)
            history.append(dict(epoch=epoch + 1, train_loss=total / train_count,
                                validation_loss=metrics['loss'], validation_f1=metrics['f1'], training_seconds=train_seconds))
            if metrics['loss'] < best:
                best = metrics['loss']
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1
            print(f'{mode} data={train_fraction:.0%} {epoch+1}/{epochs}: train={total/train_count:.4f}, validation={metrics["loss"]:.4f}', flush=True)
        synchronize(device)
        fit_seconds = time.perf_counter() - started
        model.load_state_dict(best_state)
        metrics, predictions, probabilities = evaluate(model, loaders[2], device)
        rows.append(dict(mode=mode, train_fraction=train_fraction, train_samples=train_count,
                         encoder_trainable=mode != 'frozen', max_length=max_length,
                         parameters=sum(p.numel() for p in model.parameters()),
                         trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                         training_seconds=sum(h['training_seconds'] for h in history), fit_wall_seconds=fit_seconds,
                         peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None,
                         best_epoch=best_epoch, **metrics))
        histories[mode] = history
        predictions_frame = frame.iloc[3200:].copy()
        predictions_frame['target'] = labels[3200:].tolist()
        predictions_frame['prediction'] = predictions
        predictions_frame['positive_probability'] = probabilities[:, 1]
        predictions_frame['confidence'] = probabilities.max(axis=1)
        predictions_frame.to_csv(output / f'{mode}_predictions.csv', index=False)
        pd.DataFrame(rows).to_csv(output / 'results.csv', index=False)
        (output / 'history.json').write_text(json.dumps(histories, indent=2), encoding='utf-8')
        torch.save({'state_dict': best_state, 'model_name': model_name, 'mode': mode,
                    'max_length': max_length, 'config': model.encoder.config.to_dict()}, output / f'{mode}.pt')
        del model, optimizer, best_state
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    plot_results(output, histories)
    if export_dir is not None:
        export_results(output, export_dir)
    return pd.DataFrame(rows)


def plot_results(output, histories):
    mode = 'finetuned' if 'finetuned' in histories else next(reversed(histories))
    pred = pd.read_csv(output / f'{mode}_predictions.csv')
    cm = confusion_matrix(pred.target, pred.prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Negative', 'Positive'],
           yticklabels=['Negative', 'Positive'], xlabel='Predicted', ylabel='True', title=mode)
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=180)
    plt.close(fig)
    confidence_analysis(pred, output)


def confidence_analysis(pred, output):
    """Compare actual accuracy with max-softmax confidence on held-out reviews."""
    pred = pred.copy()
    pred['correct'] = pred.target.eq(pred.prediction)
    pred['confidence_group'] = pd.cut(pred.confidence, [0.5, 0.7, 0.9, 1.0],
                                     labels=['Low [0.5, 0.7]', 'Medium (0.7, 0.9]', 'High (0.9, 1.0]'],
                                     include_lowest=True)
    summary = pred.groupby('confidence_group', observed=False).agg(
        count=('correct', 'size'), accuracy=('correct', 'mean'), mean_confidence=('confidence', 'mean'))
    summary.to_csv(output / 'confidence_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for correct, label, color in [(True, 'Correct', 'tab:blue'), (False, 'Wrong', 'tab:orange')]:
        axes[0].hist(pred.loc[pred.correct.eq(correct), 'confidence'], bins=np.linspace(.5, 1, 11),
                     alpha=.6, label=label, color=color)
    axes[0].set(xlabel='Max softmax probability', ylabel='Test examples', title='Fine-tuned BERT: confidence distribution')
    axes[0].legend()
    x = np.arange(3)
    axes[1].bar(x-.18, summary.accuracy, width=.36, label='Actual accuracy')
    axes[1].bar(x+.18, summary.mean_confidence, width=.36, label='Mean confidence')
    for i, row in enumerate(summary.itertuples()):
        axes[1].text(i, 1.03, f'n={row.count}', ha='center')
        if row.count == 0:
            axes[1].text(i, .05, 'No samples', ha='center', fontsize=9)
    axes[1].set(xticks=x, xticklabels=['Low\n[0.5, 0.7]', 'Medium\n(0.7, 0.9]', 'High\n(0.9, 1.0]'],
                ylim=(0, 1.15), ylabel='Proportion', title='Confidence does not guarantee correctness')
    axes[1].legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(output / 'confidence_analysis.png', dpi=180)
    plt.close(fig)


def run_data_efficiency(data_path=ROOT / 'datasets/imdb/IMDB Dataset.csv',
                        output_dir=ROOT / 'training_output', epochs=4, device='auto',
                        model_name=MODEL, max_length=64, batch_size=32, export_dir=ROOT):
    """Train four fresh pretrained models; no model continues from another subset."""
    output = Path(output_dir)
    rows = []
    for fraction in [.1, .25, .5, 1.0]:
        fraction_output = output / 'data_efficiency' / str(round(fraction * 100))
        result = run(data_path, fraction_output, epochs, device, model_name,
                     ('finetuned',), max_length, batch_size, train_fraction=fraction, export_dir=None)
        rows.extend(result.to_dict('records'))
        pd.DataFrame(rows).to_csv(output / 'data_efficiency.csv', index=False)
    data = pd.DataFrame(rows).sort_values('train_fraction')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for metric, label in [('accuracy', 'Accuracy'), ('f1', 'Macro F1')]:
        ax.plot(data.train_fraction * 100, data[metric], marker='o', label=label)
    ax.set(xticks=[10, 25, 50, 100], xlabel='Training data (%)', ylabel='Test score',
           ylim=(0, 1), title='Fine-tuned BERT: data efficiency (seed 42)')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'data_efficiency.png', dpi=180)
    plt.close(fig)
    if export_dir is not None:
        export_results(output, export_dir)
    return data


def export_results(output_dir, result_dir=ROOT):
    """Publish computed deliverables beside the source; checkpoints stay internal."""
    output, dest = Path(output_dir), Path(result_dir)
    dest.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(output / 'results.csv').assign(experiment='classification')
    if 'train_fraction' not in results:
        results['train_fraction'], results['train_samples'] = 1.0, 2800
    if (output / 'data_efficiency.csv').exists():
        data = pd.read_csv(output / 'data_efficiency.csv').assign(experiment='data_efficiency')
        results = pd.concat([results, data], ignore_index=True)
    results.to_csv(dest / 'results.csv', index=False)
    for filename in ['confusion_matrix.png', 'data_efficiency.png', 'confidence_analysis.png']:
        source = output / filename
        if source.exists() and source.resolve() != (dest / filename).resolve():
            shutil.copy2(source, dest / filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'datasets/imdb/IMDB Dataset.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'training_output')
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--modes', nargs='+', choices=['scratch', 'frozen', 'finetuned'], default=['scratch', 'frozen', 'finetuned'])
    parser.add_argument('--data-efficiency', action='store_true', help='Also train with 10/25/50/100 percent data.')
    args = parser.parse_args()
    run(args.data, args.output, args.epochs, args.device, args.model, args.modes, args.max_length, args.batch_size)
    if args.data_efficiency:
        run_data_efficiency(args.data, args.output, args.epochs, args.device, args.model, args.max_length, args.batch_size)
