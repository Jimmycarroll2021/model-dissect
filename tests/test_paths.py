"""Publication path isolation regressions; no model loading or inference."""
from pathlib import Path
import pytest
from minimind_re.paths import artifact_path, artifact_root
from minimind_re.transfer3.science import Evidence


def test_default_and_dedicated_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('MODEL_DISSECT_DATA', raising=False)
    assert artifact_root() == tmp_path/'.model-dissect-data'
    assert artifact_path('workspaces/new') == tmp_path/'.model-dissect-data/workspaces/new'
    root = tmp_path/'dedicated'
    monkeypatch.setenv('MODEL_DISSECT_DATA', str(root))
    assert artifact_path() == root
    evidence = Evidence(root/'packet')
    assert evidence.root == root/'packet'
    assert evidence.root.is_dir()


@pytest.mark.parametrize('value', ['', 'relative', '.', '..'])
def test_reject_relative_configuration(monkeypatch, value):
    monkeypatch.setenv('MODEL_DISSECT_DATA', value)
    with pytest.raises(ValueError, match='absolute dedicated'):
        artifact_root()


def test_reject_broad_roots_and_aliases(monkeypatch, tmp_path):
    checkout = Path(__file__).resolve().parents[1]
    for root in (Path('/'), Path.home(), checkout, checkout.parent, Path.cwd()):
        monkeypatch.setenv('MODEL_DISSECT_DATA', str(root))
        with pytest.raises(ValueError, match='dedicated artifact'):
            artifact_root()
    alias = tmp_path/'root-alias'
    alias.symlink_to('/', target_is_directory=True)
    monkeypatch.setenv('MODEL_DISSECT_DATA', str(alias))
    with pytest.raises(ValueError, match='dedicated artifact'):
        artifact_root()


@pytest.mark.parametrize('relative', ['../escape', 'safe/../../escape', '/outside'])
def test_reject_absolute_and_parent_traversal(monkeypatch, tmp_path, relative):
    monkeypatch.setenv('MODEL_DISSECT_DATA', str(tmp_path/'dedicated'))
    with pytest.raises(ValueError, match='relative.*parent traversal'):
        artifact_path(relative)


def test_resolved_symlink_containment(monkeypatch, tmp_path):
    root = tmp_path/'dedicated'
    root.mkdir()
    outside = tmp_path/'outside'
    outside.mkdir()
    (root/'escape').symlink_to(outside, target_is_directory=True)
    (root/'inside').mkdir()
    (root/'alias').symlink_to(root/'inside', target_is_directory=True)
    monkeypatch.setenv('MODEL_DISSECT_DATA', str(root))
    with pytest.raises(ValueError, match='outside'):
        artifact_path('escape/new')
    assert artifact_path('alias/new') == root/'inside/new'
    with pytest.raises(ValueError, match='bulk evidence'):
        Evidence(root/'escape/new')
    assert not (outside/'new').exists()


def test_evidence_rejects_universal_root_before_write(monkeypatch, tmp_path):
    monkeypatch.setenv('MODEL_DISSECT_DATA', '/')
    output = tmp_path/'must-not-exist'
    with pytest.raises(ValueError, match='dedicated artifact'):
        Evidence(output)
    assert not output.exists()
