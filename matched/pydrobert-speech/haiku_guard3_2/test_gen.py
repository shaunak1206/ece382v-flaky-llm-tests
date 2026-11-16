```python
"""Unit tests for the linear filter banks in pydrobert.speech.filters.

Covers the LinearFilterBank interface and its concrete subclasses
(TriangularOverlappingFilterBank, Fbank, GaborFilterBank,
ComplexGammatoneFilterBank), with a focus on get_frequency_response and
get_truncated_response. Nothing here uses random numbers, the clock, or
hash(); all comparisons on floats use tolerances. The only module attribute
modified is changed through pytest's monkeypatch fixture.
"""

import numpy as np
import pytest

from pydrobert.speech import config
from pydrobert.speech.filters import (
    ComplexGammatoneFilterBank,
    Fbank,
    GaborFilterBank,
    LinearFilterBank,
    TriangularOverlappingFilterBank,
)
from pydrobert.speech.scales import MelScaling

RATE = 16000
WIDTH = 512
HALF = WIDTH // 2 + 1  # number of half-spectrum bins for an even width