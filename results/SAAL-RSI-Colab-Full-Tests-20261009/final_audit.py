from pathlib import Path
import hashlib,json,subprocess,sys,xml.etree.ElementTree as ET
r=Path('/content/saal-tests-r3');base=r/'base';freeze=json.loads((base/'corrective-evidence/source-inventory.json').read_text());changed={n:{'original':h,'actual':hashlib.sha256((base/n).read_bytes()).hexdigest()} for n,h in freeze['files'].items() if hashlib.sha256((base/n).read_bytes()).hexdigest()!=h};assert set(changed)=={'tests/integration/test_attempt_mtls_process.py'}
check=subprocess.run([sys.executable,'-m','pip','--python','/content/saal-tests-r2/env/bin/python','check'],capture_output=True,text=True);assert check.returncode==0,check.stdout
counts={k:sum(int(s.get(k,0)) for s in ET.parse(r/'gpu-tests.xml').getroot()) for k in ['tests','failures','errors','skipped']};assert counts=={'tests':2,'failures':0,'errors':0,'skipped':0}
audit={'frozen_inventory_files':len(freeze['files']),'unchanged_files':len(freeze['files'])-len(changed),'documented_test_only_changes':changed,'production_source_changes':0,'dependency_consistency':check.stdout.strip(),'cuda_probe_counts':counts};(r/'final-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
