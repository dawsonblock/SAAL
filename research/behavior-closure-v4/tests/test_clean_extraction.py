import hashlib,importlib.util,zipfile
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('extract_capsule',Path(__file__).resolve().parents[1]/'extract_capsule.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_all_archive_members_read_into_new_directory(tmp_path):
    archive=tmp_path/'test.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('src/file.py','preserved bytes')
    h=module.digest(archive);result=module.extract(archive,tmp_path/'clean',h)
    assert result['members_read_completely']==1
    assert (tmp_path/'clean/src/file.py').read_text()=='preserved bytes'
    with pytest.raises(FileExistsError):module.extract(archive,tmp_path/'clean',h)


@pytest.mark.parametrize('name',['../escape.py','/absolute.py','src/../../escape.py','src\\escape.py'])
def test_archive_escape_rejected_before_any_member_is_written(tmp_path,name):
    archive=tmp_path/'unsafe.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr(name,'no')
    with pytest.raises(ValueError):module.extract(archive,tmp_path/'clean',module.digest(archive))
    assert not (tmp_path/'clean').exists()


def test_substituted_archive_rejected_before_extraction(tmp_path):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('source','no')
    with pytest.raises(ValueError):module.extract(archive,tmp_path/'clean','0'*64)
    assert not (tmp_path/'clean').exists()
