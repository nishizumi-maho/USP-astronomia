import numpy as np

from tccastro.openset.scores import energy_score, entropy_score


def test_entropy_score_simple() -> None:
    probs = np.array([[0.5, 0.5], [0.9, 0.1]])
    scores = entropy_score(probs)
    assert scores[0] > scores[1]


def test_energy_score_simple() -> None:
    logits = np.array([[0.0, 0.0], [2.0, -2.0]])
    scores = energy_score(logits)
    assert scores[0] > scores[1]
