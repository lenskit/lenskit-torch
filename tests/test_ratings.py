"""
Test the row-batch capabilities.
"""

from lktorch.data.ratings import RatingData

from lkbuild.data import ml_small


def test_rating_data():
    ratings = ml_small.ratings
    data = RatingData.from_ratings(ratings)

    assert data.n_samples == len(ratings)
    assert data.n_users == ratings['user'].nunique()
    assert data.n_items == ratings['item'].nunique()
