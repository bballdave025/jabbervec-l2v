from jabbervec.data import Vocab


def test_vocab_round_trip():
    vocab = Vocab.from_text("abc cab")
    text = "cab"
    assert vocab.decode(vocab.encode(text)) == text
