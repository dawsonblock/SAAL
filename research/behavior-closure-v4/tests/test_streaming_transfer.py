import hashlib
from pathlib import Path
import pytest
from saal_rsi.epoch_backend_v3 import StreamingArchive
from saal_rsi.epoch_backend_v2 import assemble_transfer


def fixtures(tmp_path):
    data=[b'first source chunk',b'second source chunk',b'last source chunk']
    chunks=[]
    for i,blob in enumerate(data):
        name=f'part-{i:03d}.bin';(tmp_path/name).write_bytes(blob)
        chunks.append({'name':name,'size':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
    manifest={'schema':'saal-artifact-transfer-v1','job_digest':'sha256:'+'1'*64,
        'size':sum(map(len,data)),'sha256':hashlib.sha256(b''.join(data)).hexdigest(),'chunks':chunks}
    return data,manifest


def test_streamed_archive_matches_original_with_only_one_copy(tmp_path):
    data,manifest=fixtures(tmp_path);old=tmp_path/'old.zip'
    assemble_transfer(tmp_path,manifest,old)
    new=tmp_path/'stream.zip';sink=StreamingArchive(new,manifest)
    for i,chunk in enumerate(manifest['chunks']):
        p=tmp_path/chunk['name'];sink.append(p,i);p.unlink()
        assert sink.written==sum(map(len,data[:i+1]))
    sink.finish();sink.close()
    assert new.read_bytes()==old.read_bytes()==b''.join(data)
    assert not list(tmp_path.glob('part-*.bin'))


@pytest.mark.parametrize('fault',['reordered','corrupted','incomplete','duplicate','disk_substitution'])
def test_invalid_stream_cannot_complete(tmp_path,fault):
    data,manifest=fixtures(tmp_path);target=tmp_path/'stream.zip';sink=StreamingArchive(target,manifest)
    try:
        with pytest.raises(ValueError):
            if fault=='reordered':sink.append(tmp_path/'part-001.bin',1)
            elif fault=='corrupted':
                (tmp_path/'part-000.bin').write_bytes(b'x'*len(data[0]));sink.append(tmp_path/'part-000.bin',0)
            elif fault=='incomplete':sink.append(tmp_path/'part-000.bin',0);sink.finish()
            elif fault=='duplicate':sink.append(tmp_path/'part-000.bin',0);sink.append(tmp_path/'part-000.bin',0)
            else:
                for i,c in enumerate(manifest['chunks']):sink.append(tmp_path/c['name'],i)
                sink.file.flush();target.write_bytes(b'x'*manifest['size']);sink.finish()
        assert not sink.finished
    finally:sink.close()
