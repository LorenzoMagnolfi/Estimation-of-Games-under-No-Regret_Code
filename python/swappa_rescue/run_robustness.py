"""Predeclared robustness queue; never labels sensitivity budgets as confidence."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from audit_panel import ROOT, OUT

CASES = [
    ['--nooffer','--support-high','2000'],
    ['--nooffer','--individual-ir','--price-step','5','--cost-step','10'],
    ['--nooffer','--individual-ir','--outcome','sale_forward'],
    ['--nooffer','--individual-ir','--outcome','sale_legacy'],
    ['--nooffer','--individual-ir','--spec','mean'],
    ['--nooffer','--individual-ir','--spec','min'],
]

def main():
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    manifest={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'working_tree':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip(),'source_sha256':sources,'cases':CASES,'results':[],'status':'running','interpretation':'Exploratory robustness. No first-stage or behavioral-envelope coverage certified.'}
    path=OUT/'robustness_manifest.json'
    path.write_text(json.dumps(manifest,indent=2))
    for i,args in enumerate(CASES):
        logfile=OUT/f'robustness_{i:02}.log'
        print(f'Starting case {i}: {args}',flush=True)
        started=datetime.datetime.now(datetime.timezone.utc).isoformat()
        with logfile.open('w') as log:
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('continuous_cost_lp.py')),*args],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        manifest['results'].append({'case':i,'args':args,'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':result.returncode,'log':str(logfile.relative_to(ROOT))})
        path.write_text(json.dumps(manifest,indent=2))
        print(f'Case {i} exited {result.returncode}',flush=True)
        if result.returncode:
            manifest['status']='failed'; path.write_text(json.dumps(manifest,indent=2)); return result.returncode
    manifest['status']='completed'; path.write_text(json.dumps(manifest,indent=2)); return 0

if __name__=='__main__':sys.exit(main())
