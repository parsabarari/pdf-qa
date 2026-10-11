
import pytest
from pydantic import ValidationError

from app.core.config import Settings


@pytest.mark.parametrize("score", [-1.0, 1.0, 0.25])
def test_min_relevance_score_accepts_valid_values(score):
    settings = Settings(min_relevance_score=score)

    assert settings.min_relevance_score == score


@pytest.mark.parametrize("score", [-1.1, 1.1, 1.5, -2.0])
def test_min_relevance_score_rejects_invalid_values(score):
    with pytest.raises(ValidationError):
        Settings(min_relevance_score=score)
