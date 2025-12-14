import os
import pytest
import torch

from src.model import get_model, get_tokenizer, MODEL_NAME, NUM_LABELS


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skip heavy model download in CI"
)
def test_model_instantiation():
    """
    Ensure the model can be instantiated with the correct number of labels.
    Skips if the model cannot be downloaded in this environment.
    """
    try:
        model = get_model(MODEL_NAME, NUM_LABELS)
    except OSError:
        pytest.skip("Model cannot be downloaded in this environment.")

    assert model is not None
    assert model.config.num_labels == NUM_LABELS


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skip heavy model forward pass in CI"
)
def test_model_forward_pass():
    """
    Run a small forward pass and check the logits shape.
    Skips if the model/tokenizer cannot be downloaded in this environment.
    """
    try:
        tokenizer = get_tokenizer(MODEL_NAME)
        model = get_model(MODEL_NAME, NUM_LABELS)
    except OSError:
        pytest.skip("Model cannot be downloaded in this environment.")

    texts = ["i love this", "this is bad"]
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**encodings)

    logits = outputs.logits
    assert logits.shape[0] == len(texts)
    assert logits.shape[1] == NUM_LABELS
