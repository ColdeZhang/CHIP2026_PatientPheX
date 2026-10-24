"""Distinct immutable launch-validity and shared inference-budget clocks."""
import math
import time


def validate(plan):
    a = plan['admission']
    assert a['schema'] == 'training-seed-F1-admission-v1'
    for key in ('bound_at_unix', 'launch_not_after_unix', 'deadline_TERM_unix', 'deadline_KILL_unix'):
        assert type(a[key]) in (int, float) and math.isfinite(a[key])
    assert plan['budget_seconds'] == plan['authorized_budget_seconds'] == 1800
    assert plan['grace_seconds'] == 30
    assert 0 < a['launch_not_after_unix'] - a['bound_at_unix'] <= 300
    assert a['deadline_TERM_unix'] == a['bound_at_unix'] + 1800
    assert a['deadline_KILL_unix'] == a['deadline_TERM_unix'] + 30
    return a


def remaining(plan, now=None):
    return validate(plan)['deadline_TERM_unix'] - (time.time() if now is None else now)


def admit(plan, stage, now=None):
    a = validate(plan)
    now = time.time() if now is None else now
    assert a['bound_at_unix'] <= now <= a['launch_not_after_unix'], (
        stage + ': immutable launch validity expired; preserve intent, do not retry')
    assert remaining(plan, now) > 0, stage + ': shared inference deadline expired'
    return a
