from datetime import date, timedelta

from ensemble.data.splits import make_splits


def test_weeks_are_contiguous_and_disjoint():
    s = make_splits(date(2020, 9, 22))
    assert s.test.start == date(2020, 9, 16) and s.test.end == date(2020, 9, 22)
    assert s.val.end + timedelta(days=1) == s.test.start
    assert (s.val.end - s.val.start).days == 6
    assert s.submission.start == date(2020, 9, 23)


def test_cutoff_precedes_label_week():
    s = make_splits(date(2020, 9, 22))
    for w in [s.val, s.test, *s.train_weeks(4)]:
        assert w.cutoff < w.start


def test_train_weeks_precede_validation():
    s = make_splits(date(2020, 9, 22))
    weeks = s.train_weeks(3)
    assert weeks[-1].end < s.val.start
    assert [w.start for w in weeks] == sorted(w.start for w in weeks)
