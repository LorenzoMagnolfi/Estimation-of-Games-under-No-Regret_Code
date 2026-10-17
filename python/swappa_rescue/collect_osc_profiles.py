"""Collect only completed Swappa-owned OSC tasks through verified SFTP."""
import argparse,datetime,hashlib,json,shlex,subprocess
from pathlib import Path
import pandas as pd
from audit_panel import OUT


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',default='swappa_1e112e66_20261016');args=ap.parse_args();campaign=OUT/'osc_campaign'/args.run_id;mp=campaign/'manifest.json';m=json.loads(mp.read_text());remote=m['remote_directory'];assert remote.startswith('/users/PAS3582/magnolfi1/swappa/runs/swappa_');now=datetime.datetime.now(datetime.timezone.utc)
 if m.get('last_collection_UTC') and (now-datetime.datetime.fromisoformat(m['last_collection_UTC'])).total_seconds()<300:
  print('Collection skipped: five-minute monitoring interval has not elapsed.');return
 common=['-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-i',str(Path.home()/'.ssh/id_ed25519_osc')];ssh=['ssh','-T',*common,'-o','UserKnownHostsFile='+str(Path.home()/'.ssh/known_hosts_osc'),'magnolfi1@cardinal.osc.edu']
 code="import pathlib,json; root=pathlib.Path("+repr(remote)+"); print(json.dumps([json.loads(p.read_text()) for p in root.glob('results/task_*/status.json')]))"
 r=subprocess.run(ssh+['/users/PAS3582/magnolfi1/envs/pc3/bin/python -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=60);statuses=json.loads(r.stdout);done=set(m.get('downloaded_tasks',[]));commands=[];pending=[]
 for status in statuses:
  task=int(status['task'])
  if task==0 or task in done:continue
  local=campaign/'downloads'/f't{task:04d}';local.mkdir(parents=True,exist_ok=True);mapping={}
  for entry in status['files']:
   alias='bounds.csv' if entry['name'].endswith('.csv') else 'coefficients.npz';mapping[entry['name']]=alias
   commands.append('get "'+remote+'/results/task_'+str(task)+'/'+entry['name']+'" "'+(local/alias).as_posix()+'"')
  for name in ['status.json','stdout.txt','stderr.txt']:commands.append('get "'+remote+'/results/task_'+str(task)+'/'+name+'" "'+(local/name).as_posix()+'"')
  pending.append((status,local,mapping))
 if commands:
  batch=campaign/'download_batch.txt';batch.write_text(chr(10).join(commands)+chr(10));cmd=['sftp','-b',str(batch),*common,'-o','UserKnownHostsFile='+str(OUT/'known_hosts_transfer_verified'),'magnolfi1@sftp.osc.edu'];r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
  if r.returncode:raise RuntimeError(r.stderr[-2000:])
  for status,local,mapping in pending:
   downloaded=json.loads((local/'status.json').read_text());assert downloaded==status and status['code_commit']==m['code_commit']
   for entry in status['files']:assert hashlib.sha256((local/mapping[entry['name']]).read_bytes()).hexdigest()==entry['sha256']
   (local/'file_aliases.json').write_text(json.dumps(mapping,indent=2));done.add(status['task'])
 m['downloaded_tasks']=sorted(done);m['last_collection_UTC']=now.isoformat();mp.write_text(json.dumps(m,indent=2));records=[];taskdefs=json.loads((campaign/'tasks.json').read_text());statusmap={x['task']:x for x in statuses}
 for delta in [20,35,50]:
  expected=[i for i,t in enumerate(taskdefs) if i>0 and t['delta']==delta];complete=all(i in done and statusmap[i]['returncode']==0 for i in expected)
  if not complete:continue
  frames=[]
  for task in expected:
   p=campaign/'downloads'/f't{task:04d}'/'bounds.csv'
   if p.exists():d=pd.read_csv(p);d['task']=task;frames.append(d)
   else:assert statusmap[task]['infeasible_prior']
  d=pd.concat(frames,ignore_index=True)
  assert d.groupby(['task','copies']).solver_status.nunique().max()==1,'Contradictory objective feasibility statuses; resolve before aggregation'
  for (copies,estimand),group in d.groupby(['copies','estimand']):
   assert set(group.solver_status.unique())<=set([0,2]),'Unresolved solver failure; do not report a global interval'
   accepted=group[group.success];low=accepted.loc[accepted.direction.eq('low'),'cost_bound'];high=accepted.loc[accepted.direction.eq('high'),'cost_bound'];price=float(group.mean_observed_price.iloc[0]);cl=float(low.min()) if len(low) else None;ch=float(high.max()) if len(high) else None
   records.append({'run_id':args.run_id,'code_commit':m['code_commit'],'model':str(group.model.iloc[0]),'critical':float(group.critical.iloc[0]),'alpha_R':float(group.alpha_R.iloc[0]),'cells':int(group.cells.iloc[0]),'delta':delta,'copies':int(copies),'estimand':estimand,'all_rho_boxes_accounted_for':True,'accepted_boxes':int(accepted.task.nunique()),'cost_low':cl,'cost_high':ch,'gross_margin_low':price-ch if ch is not None else None,'gross_margin_high':price-cl if cl is not None else None,'prior_cost_low':float(group.prior_cost_low.min()),'prior_cost_high':float(group.prior_cost_high.max()),'mean_price':price})
 if records:pd.DataFrame(records).to_csv(campaign/'global_profile_bounds.csv',index=False)
 summary={'downloaded_profile_tasks':len(done),'expected_profile_tasks':180,'complete_deltas':sorted(set(x['delta'] for x in records)),'failed_tasks':[x['task'] for x in statuses if x['returncode']!=0]};print(json.dumps(summary),flush=True)
 if len(done)==180 and not summary['failed_tasks']:m['status']='profile_results_collected';mp.write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
