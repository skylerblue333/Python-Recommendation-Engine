from src.engine import CollaborativeFilter


def test_training_and_recommendation():
    cf = CollaborativeFilter(num_users=5, num_items=10)
    interactions = [
        (0, 1, 5.0),
        (0, 2, 4.0),
        (1, 1, 4.5),
        (1, 3, 5.0),
        (2, 4, 1.0),
        (2, 5, 2.0),
    ]
    cf.train(interactions, epochs=5)
    recos = cf.recommend(0, top_k=3)
    assert len(recos) == 3
