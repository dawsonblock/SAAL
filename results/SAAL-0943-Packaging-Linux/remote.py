from pathlib import Path
import subprocess,sys,os,json,zipfile,hashlib,tempfile,platform
r=Path('/content/package-tests');r.mkdir(exist_ok=True);out=r/'evidence';out.mkdir(exist_ok=True);report={'status':'RUNNING','runtime':{'python':sys.version,'platform':platform.platform()}}
try:
 with zipfile.ZipFile('/content/package-input.zip') as z:z.extractall(r)
 for n,sha in json.loads((r/'HASHES.json').read_text()).items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==sha
 subprocess.run([sys.executable,'-m','venv','--without-pip',str(r/'env')],check=True,timeout=60);python=str(r/'env/bin/python')
 with (out/'install.log').open('w') as f:subprocess.run([sys.executable,'-m','pip','--python',python,'install','--require-hashes','--only-binary=:all:','-r',str(r/'locked.txt')],check=True,stdout=f,stderr=subprocess.STDOUT,timeout=240)
 with (out/'tests.log').open('w') as f:subprocess.run([python,'-m','pytest','-q','--junitxml='+str(out/'tests.xml'),str(r/'tests/reproducibility')],check=True,stdout=f,stderr=subprocess.STDOUT,timeout=60)
 code="from pathlib import Path;from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey;import tempfile,hashlib,json,subprocess,sys,zipfile\nwith tempfile.TemporaryDirectory() as t:\n t=Path(t);root=t/'release';root.mkdir();e=root/'reproducibility-evidence';e.mkdir();(root/'artifact').write_bytes(b'physical checkpoint');(e/'parent-inventory.json').write_text(json.dumps({'commit':'fixture','files':{'artifact':hashlib.sha256((root/'artifact').read_bytes()).hexdigest()}}));key=t/'key';key.write_bytes(Ed25519PrivateKey.generate().private_bytes_raw());subprocess.run([sys.executable,'/content/package-tests/tools/build_reproducible_release.py','--root',str(root),'--private-key',str(key),'--output',str(t/'out.zip')],check=True);inside=root/'private';inside.write_bytes(key.read_bytes());p=subprocess.run([sys.executable,'/content/package-tests/tools/build_reproducible_release.py','--root',str(root),'--private-key',str(inside)],capture_output=True,text=True);assert p.returncode!=0 and 'outside checkout' in p.stderr;print('signed roundtrip and private-key boundary PASS')\n"
 with (out/'roundtrip.log').open('w') as f:subprocess.run([python,'-c',code],check=True,stdout=f,stderr=subprocess.STDOUT,timeout=60)
 report.update(status='PASS',source_hashes=json.loads((r/'HASHES.json').read_text()),lock_sha256=hashlib.sha256((r/'locked.txt').read_bytes()).hexdigest())
except Exception as ex:report.update(status='FAIL',error=repr(ex))
finally:
 (out/'REPORT.json').write_text(json.dumps(report,indent=2))
 with zipfile.ZipFile('/content/packaging-evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in out.iterdir():z.write(p,p.name)
 print(json.dumps(report,indent=2))
