from pytest import fixture
from pytorch_lightning import seed_everything
from transformers import set_seed


@fixture
def seed():
    seed_everything(0)
    set_seed(0)
