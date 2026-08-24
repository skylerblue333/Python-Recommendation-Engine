import pytest

from src.engine import CollaborativeFilter


def interactions() -> list[tuple[int, int, float]]:
    return [(0, 1, 5.0), (0, 2, 4.0), (1, 1, 4.5), (1, 3, 5.0), (2, 4, 1.0), (2, 5, 2.0)]


def test_training_is_deterministic_and_excludes_seen_items() -> None:
    first = CollaborativeFilter(num_users=5, num_items=10, seed=4444)
    second = CollaborativeFilter(num_users=5, num_items=10, seed=4444)
    result = first.train(interactions(), epochs=5)
    second.train(interactions(), epochs=5)
    assert result.interactions == 6
    assert result.final_mse >= 0
    assert first.recommend(0, top_k=3) == second.recommend(0, top_k=3)
    assert 1 not in first.recommend(0, top_k=8)
    assert 2 not in first.recommend(0, top_k=8)


def test_invalid_inputs_fail_closed() -> None:
    engine = CollaborativeFilter(num_users=2, num_items=3)
    with pytest.raises(ValueError):
        engine.train([])
    with pytest.raises(ValueError):
        engine.train([(2, 0, 1.0)])
    with pytest.raises(ValueError):
        engine.recommend(0, top_k=0)
    with pytest.raises(ValueError):
        engine.score(0, [3])
