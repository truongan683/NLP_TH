"""Answer an English question from a local UTF-8 text document."""
import argparse
from pathlib import Path

import torch

from qa_model import Example, QAModel, evaluate, load_tokenizer, make_features, make_loader


def answer_document(question, context, checkpoint, device):
    if not question.strip() or not context.strip():
        raise ValueError("Question and document must be nonempty")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tokenizer = load_tokenizer(saved["backbone"])
    model = QAModel(saved["backbone"]).to(device)
    model.load_state_dict(saved["state_dict"])
    features = make_features([Example("document", question, context)], tokenizer, saved["max_length"], saved["stride"])
    _, predictions = evaluate(model, make_loader(features, tokenizer, 16), 1, device)
    return predictions[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("question")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "training_output" / "lab10" / "qa.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        context = args.document.read_text(encoding="utf-8-sig")
        span = answer_document(args.question, context, args.checkpoint, device)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(span.text(context))
    print(f"Character span: [{span.start}, {span.end})")


if __name__ == "__main__":
    main()
