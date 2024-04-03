from mim_nlp.seq2seq import Seq2SeqDataset


def test_seq2seq_dataset():
    dataset = Seq2SeqDataset(["abc", "def"], y_train=["a", "b"], source_transform_fn=lambda x: x[:2])
    assert dataset[0].source == "ab"
    assert dataset[1].target == "b"
