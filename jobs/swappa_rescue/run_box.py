"""Run one frozen-demand profile task and export its owned results."""
import argparse,hashlib,json,os,subprocess,sys,time,shutil
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);ap.add_argument('--tasks',required=True);ap.add_argument('--export',required=True);args=ap.parse_args();root=Path(__file__).resolve().parents[2];out=root/'output/swappa_rescue_20261016';out.mkdir(parents=True,exist_ok=True);spec=json.loads(Path(args.tasks).read_text())[args.task];export=Path(args.export);export.mkdir(parents=True,exist_ok=True)
 cmd=[sys.executable,str(root/'python/swappa_rescue/normalized_nonparam.py'),'--bundle',str(root/'inputs/upgrade_normalized_bundle'),'--rho-low',str(spec['rho_low']),'--rho-high',str(spec['rho_high']),'--delta',str(spec['delta']),'--cells',str(spec['cells']),'--grid-points','33'];start=time.time();r=subprocess.run(cmd,cwd=root,capture_output=True,text=True);(export/'stdout.txt').write_text(r.stdout);(export/'stderr.txt').write_text(r.stderr);files=[]
 for p in out.glob('normalized_nonparam*'):
  if p.is_file():shutil.copy2(p,export/p.name);files.append({'name':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 prior_infeasible=any(json.loads(x).get('infeasible_prior',False) for x in r.stdout.splitlines() if x.startswith('{'))
 status={'task':args.task,'spec':spec,'returncode':r.returncode,'seconds':time.time()-start,'job_id':os.getenv('SLURM_JOB_ID'),'code_commit':(root/'CODE_COMMIT').read_text().strip(),'infeasible_prior':prior_infeasible,'files':files};(export/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 assert prior_infeasible or len(files)==2,'Missing inference outputs'
if __name__=='__main__':main()
