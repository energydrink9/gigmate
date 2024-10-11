from gigmate.model.codec import get_codec


def test_codec() -> None:
    get_codec('cpu')
    