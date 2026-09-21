import numpy as np
import pandas as pd
import pytest
from robird import explanatory_suite_v1_1 as a
from robird.io import atomic_write_csv


def real_fixture():
    return pd.DataFrame([dict(observation_id=o, observer_id=o+10, photo_id=o*10+j,
        taxon_id=1, class_index=0, difficulty=np.float32(3.3788753+j*.22+o*.01),
        difficulty_bin=j, split='development_test') for o in range(1, 5) for j in range(2)])


def csv_fixture(tmp_path, matched=True):
    real = real_fixture()
    pseudo, _ = a.legacy.regroup(real, 19, matched)
    # Original issue: float32 shortest decimal versus Python float promoted by to_dict.
    atomic_write_csv(tmp_path/'real.csv', real)
    atomic_write_csv(tmp_path/'pseudo.csv', pseudo)
    return pd.read_csv(tmp_path/'real.csv'), pd.read_csv(tmp_path/'pseudo.csv')


@pytest.mark.parametrize('matched', [False, True])
def test_real_csv_roundtrip_original_fails_adapter_passes(tmp_path, matched):
    real, pseudo = csv_fixture(tmp_path, matched)
    with pytest.raises(a.GateStop, match='difficulty'):
        a._original_validate(real, pseudo, matched)
    assert a.validate_mapping(real, pseudo, matched)


@pytest.mark.parametrize('corruption', ['next_float32', 'nan', 'bin', 'photo', 'label'])
def test_serialization_adapter_still_rejects_corruption(tmp_path, corruption):
    real, pseudo = csv_fixture(tmp_path)
    if corruption == 'next_float32':
        pseudo.loc[0, 'difficulty'] = float(np.nextafter(np.float32(pseudo.loc[0, 'difficulty']), np.float32(np.inf)))
    elif corruption == 'nan': pseudo.loc[0, 'difficulty'] = np.nan
    elif corruption == 'bin': pseudo.loc[0, 'difficulty_bin'] += 1
    elif corruption == 'photo': pseudo.loc[0, 'photo_id'] = pseudo.loc[1, 'photo_id']
    else: pseudo.loc[0, 'class_index'] += 1
    with pytest.raises(a.GateStop): a.validate_mapping(real, pseudo, True)


def test_legacy_module_instance_not_globally_patched():
    from robird import explanatory_suite_v1 as original
    assert original.validate_mapping is not a.validate_mapping
    assert a.legacy.validate_mapping is a.validate_mapping


def test_copy_preserves_bytes_refuses_overwrite(tmp_path):
    source, dest = tmp_path/'src.csv', tmp_path/'dest.csv'
    atomic_write_csv(source, real_fixture())
    a.copy_exact(source, dest); a.copy_exact(source, dest)
    assert source.read_bytes() == dest.read_bytes()
    atomic_write_csv(source, real_fixture().assign(difficulty=0))
    with pytest.raises(a.GateStop): a.copy_exact(source, dest)
