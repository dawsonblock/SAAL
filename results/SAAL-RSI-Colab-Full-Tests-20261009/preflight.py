import sys,platform,json,subprocess
print(json.dumps({'python':sys.version,'platform':platform.platform(),'machine':platform.machine()}))
print(subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout)
