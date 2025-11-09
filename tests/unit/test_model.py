import pytest
pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.model import get_model, get_tokenizer, SimpleTextDataset

TINY = "hf-internal-testing/tiny-random-distilbert"

def test_tiny_model_forward():
    tok = get_tokenizer(TINY)
    enc = tok(["good movie", "bad movie"], truncation=True, padding=True)
    ds = SimpleTextDataset(enc, [1, 0])

    model = get_model(TINY, num_labels=2)
    batch = {k: v[:1] for k, v in ds[0].items()}  # one sample
    out = model(input_ids=batch["input_ids"].unsqueeze(0),
                attention_mask=batch["attention_mask"].unsqueeze(0),
                labels=batch["labels"].unsqueeze(0))
    assert out.logits.shape[-1] == 2
