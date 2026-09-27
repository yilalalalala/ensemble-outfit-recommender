"""Temporal splits (D-003).

The test week is the final 7 days of data, validation the 7 days before it.
Everything is derived from the data's max date, so the same code produces the
Kaggle submission split by shifting one week forward.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Week:
    """A 7-day label window [start, end]. Features must use dates < start."""

    start: date
    end: date

    @property
    def cutoff(self) -> date:
        """Last date whose data may be used when predicting this week."""
        return self.start - timedelta(days=1)

    def shift(self, weeks: int) -> "Week":
        d = timedelta(days=7 * weeks)
        return Week(self.start + d, self.end + d)


@dataclass(frozen=True)
class Splits:
    test: Week
    val: Week
    window_weeks: int

    @property
    def submission(self) -> Week:
        """The Kaggle private leaderboard week, right after the data ends."""
        return self.test.shift(1)

    def train_weeks(self, n: int, before: Week | None = None) -> list[Week]:
        """The ``n`` label weeks immediately preceding ``before`` (default: val)."""
        before = before or self.val
        return [before.shift(-k) for k in range(n, 0, -1)]

    def window_start(self, week: Week) -> date:
        """First date of the feature/retrieval window for a label week."""
        return week.start - timedelta(days=7 * self.window_weeks)


def make_splits(max_date: date, window_weeks: int = 16) -> Splits:
    test = Week(max_date - timedelta(days=6), max_date)
    return Splits(test=test, val=test.shift(-1), window_weeks=window_weeks)


def load_splits(con, cfg) -> Splits:
    max_date = con.execute("SELECT max(t_dat) FROM transactions").fetchone()[0]
    return make_splits(max_date, int(cfg.splits.window_weeks))
