from models.metric import mlm_metric


def test_mlm_metric():
    score = mlm_metric('How do I change my password?', 'You do not.')
    assert len(score.shape) == 0
