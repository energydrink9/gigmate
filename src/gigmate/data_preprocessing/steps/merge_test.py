

import random
from gigmate.data_preprocessing.steps.merge import create_stems_assortments, get_stem
from gigmate.utils.constants import get_random_seed

CURRENT_STEM_FILE = 'current'


def test_create_stems_assortments() -> None:
    
    random.seed(get_random_seed())

    other_stems = [
        get_stem('first_gtr', False),
        get_stem('second_bass', False),
        get_stem('second_bass', False),
        get_stem('third_silent', True),
        get_stem('forth_silent', True),
        get_stem('fifth', False),
    ]
    assortments = create_stems_assortments(other_stems, CURRENT_STEM_FILE)
    
    all_stems_assortment = (CURRENT_STEM_FILE, frozenset({'third_silent', 'forth_silent', 'second_bass', 'first_gtr', 'fifth'}))
    bass_assortment = (CURRENT_STEM_FILE, frozenset({'second_bass'}))
    gtr_assortment = (CURRENT_STEM_FILE, frozenset({'first_gtr'}))

    assert all_stems_assortment in assortments
    assert bass_assortment in assortments and gtr_assortment in assortments
    assert len(assortments) > 1


def test_create_stems_assortments_no_basic_stems() -> None:
    
    random.seed(get_random_seed())

    other_stems = [
        get_stem('first', False),
        get_stem('second', False),
        get_stem('second', False),
        get_stem('third_silent', True),
        get_stem('forth_silent', True),
        get_stem('fifth', False),
    ]
    assortments = create_stems_assortments(other_stems, CURRENT_STEM_FILE)

    expected = [
        (CURRENT_STEM_FILE, frozenset({'third_silent', 'forth_silent', 'second', 'first', 'fifth'})),
    ]

    assert sorted(assortments) == sorted(expected)


def test_create_stems_assortments_all_silent() -> None:
    
    random.seed(get_random_seed())

    other_stems = [
        get_stem('first_gtr', True),
        get_stem('second_bass', True),
        get_stem('second_bass', True),
        get_stem('third_silent', True),
        get_stem('forth_silent', True),
        get_stem('fifth', True),
    ]
    assortments = create_stems_assortments(other_stems, CURRENT_STEM_FILE)

    expected = [
        (CURRENT_STEM_FILE, frozenset({'third_silent', 'forth_silent', 'second_bass', 'first_gtr', 'fifth'})),
    ]

    assert sorted(assortments) == sorted(expected)