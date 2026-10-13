from pathlib import Path
import subprocess,os,sys,json,hashlib,xml.etree.ElementTree as ET,time,shutil
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(exist_ok=False)
py='/private/tmp/saal094-env/bin/python'
suites=[('core',['tests'],['src']),('corpus',['reviewed-decision-corpus/tests'],['src','reviewed-decision-corpus/src']),('reporting',['research/reliable-learning-v3/tests'],['src']),('project',['research/behavior-closure-v4/tests'],['src']),('diverse',['research/diverse-learning-v1/tests'],['src','research/diverse-learning-v1/src']),('reliability',['research/learning-reliability-v2/tools/test_postprocessing.py'],['src'])]
if len(sys.argv)>3:
 assert sys.argv[3]=='remaining'
 suites=suites[3:]
rows=[]
for name,tests,paths in suites:
 env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(root/p) for p in paths),PYTHONDONTWRITEBYTECODE='1')
 log=out/(name+'.log');xml=out/(name+'.xml');start=time.monotonic()
 with log.open('xb') as f:r=subprocess.run([py,'-m','pytest','-q',*tests,'--junitxml='+str(xml),'--basetemp='+str(out/(name+'-temp'))],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 if not xml.exists():raise RuntimeError('no actual XML '+name)
 tree=ET.parse(xml).getroot();ss=list(tree.iter('testsuite'))
 counts={k:sum(int(s.attrib.get(k,0)) for s in ss) for k in ('tests','failures','errors','skipped')}
 record={'suite':name,'returncode':r.returncode,**counts,'elapsed_seconds':time.monotonic()-start,'xml_sha256':hashlib.sha256(xml.read_bytes()).hexdigest(),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
 rows.append(record);print(json.dumps(record),flush=True)
 with (out/'test-results.json').open('w') as f:json.dump({'suites':rows,'complete':len(rows)==len(suites),'passed':all(x['returncode']==0 and x['failures']==0 and x['errors']==0 for x in rows)},f,indent=2)
 if r.returncode:raise RuntimeError('clean suite failed '+name)
 shutil.rmtree(out/(name+'-temp'),ignore_errors=False) if (out/(name+'-temp')).exists() else None
