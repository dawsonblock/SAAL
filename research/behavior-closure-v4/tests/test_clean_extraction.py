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


def test_hardlink_reuse_still_authenticates_the_archives_actual_tensor_bytes(tmp_path):
    cache=tmp_path/'cache';cache.mkdir();source=cache/'full.safetensors'
    with source.open('wb') as f:f.truncate(50_000_001)
    h=module.digest(source);name='tensor_payloads/'+h+'.safetensors';archive=tmp_path/'tensor.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:z.write(source,name)
    result=module.extract(archive,tmp_path/'clean',module.digest(archive),[cache])
    assert result['verified_tensor_hardlinks']==[name]
    assert (tmp_path/'clean'/name).stat().st_ino==source.stat().st_ino
    # A self-consistent archive hash does not excuse a substituted content-addressed member.
    bad=tmp_path/'substitution.zip'
    with zipfile.ZipFile(bad,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        with z.open(name,'w') as f:
            for _ in range(50):f.write(b'x'*1_000_000)
            f.write(b'x')
    with pytest.raises(ValueError):module.extract(bad,tmp_path/'bad-clean',module.digest(bad),[cache])
