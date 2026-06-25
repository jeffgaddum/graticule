from __future__ import annotations

import numpy as np

from graticule.cartouche import DataSpec
from graticule.gazetteer import synthesize_panel


def test_synthesis_is_reproducible() -> None:
    spec = DataSpec(n_units=6, n_time=5, n_cov=7, n_families=4)
    one = synthesize_panel(np.random.default_rng(123), spec, family=2)
    two = synthesize_panel(np.random.default_rng(123), spec, family=2)
    for key in one:
        assert np.array_equal(one[key], two[key])


def test_potential_outcomes_define_effect() -> None:
    spec = DataSpec(n_units=8, n_time=6, n_cov=5, signal=1.5)
    arrays = synthesize_panel(np.random.default_rng(7), spec, family=0)
    recovered = arrays["y1"] - arrays["y0"]
    assert np.allclose(recovered, arrays["tau"])
    assert arrays["tau"].std() > 0.0
