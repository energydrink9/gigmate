import os
from typing import Literal, cast

ENVIRONMENT = cast(Literal['dev', 'test', 'prod'], os.environ['ENV']) if 'ENV' in os.environ else 'dev'


def get_environment() -> Literal['dev', 'test', 'prod']:
    return ENVIRONMENT