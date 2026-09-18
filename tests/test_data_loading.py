from ml.config import TARGET
from ml.data import clean_dataset, generate_dataset, load_dataset


def test_generate_dataset_schema():
    df = generate_dataset()
    assert len(df) == 569
    assert df.shape[1] == 33  # id + diagnosis + 30 features + Unnamed: 32
    assert set(df[TARGET].unique()) == {"M", "B"}


def test_load_dataset_regenerates_when_missing(tmp_path):
    path = tmp_path / "data.csv"
    assert not path.exists()
    df = load_dataset(path=path)
    assert path.exists()
    assert len(df) > 0


def test_clean_dataset_encodes_target_and_drops_artifacts(clean_df):
    assert set(clean_df[TARGET].unique()) == {0, 1}
    assert "id" not in clean_df.columns
    assert "Unnamed: 32" not in clean_df.columns
    assert clean_df.isna().sum().sum() == 0
