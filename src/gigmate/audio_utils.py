
import random


def generate_random_filename(prefix: str = 'temp_', extension: str = '.wav') -> str:
    return f'output/{prefix}{random.randint(0, 1000000)}{extension}'