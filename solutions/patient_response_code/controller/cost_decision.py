"""Measure the unique pressure terminal and estimate the fixed two-epoch cost; no model."""
from pathlib import Path
import argparse,hashlib,json

def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--terminal',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();D=a.terminal
 w=json.loads((D/'wrapper-complete.json').read_text());s=json.loads((D/'step.json').read_text());done=json.loads((D/'complete.json').read_text())
 assert w['returncode']==0 and not w['timed_out'] and w['all12_backwards_and_one_update_verified'] and w['target_has_no_process']
 assert s['actual_forwards']==s['actual_backwards']==12 and s['actual_optimizer_steps']==1 and s['max_supervision128_and_longest5910_completed']
 assert s['final_inventory']['all_grad_finite'] and s['final_inventory']['frozen_base_no_gradient']
 assert done['step_sha256']==ref(D/'step.json')['sha256'] and done['fresh_state_discarded']
 pressure=s['diagnostic_window_seconds'];overhead=max(0.,done['native_seconds']-pressure)
 row_projection=pressure*(124*2/12);token_projection=pressure*(427560*2/44582)
 train_proxy=max(row_projection,token_projection)*1.15
 # Existing measured A800->.100 520MB ACK was56-60s; current A100 private link
 # has not been measured. Combined600s reserves both transfers and controller/queue
 # overhead, not an assertion that max600 each can both be exhausted within cap.
 export_and_control_allowance=600.
 estimate=train_proxy+2*overhead+export_and_control_allowance
 result=dict(status='ACTUAL_PRESSURE_PASS_FIXED_FORMAL_COST_ESTIMATE',source=ref(__file__),probe_wrapper=ref(D/'wrapper-complete.json'),probe_step=ref(D/'step.json'),probe_complete=ref(D/'complete.json'),pressure_seconds=pressure,observed_other_native_seconds=overhead,observed_wrapper_seconds=w['elapsed_seconds'],row_count_projection_seconds=row_projection,token_projection_seconds=token_projection,conservative_training_proxy_seconds=train_proxy,loading_and_asset_overhead_twice=2*overhead,combined_export_controller_allowance_seconds=export_and_control_allowance,estimated_complete_campaign_seconds=estimate,hard_shared_budget_seconds=3600,formal_cost_fits_3600=estimate<3600,peak_allocated_bytes=s['combined_peak_allocated_bytes'],peak_reserved_bytes=s['combined_peak_reserved_bytes'],estimated_private_SHM_bytes=4*1024**3,estimated_persistent_two_checkpoints_bytes=1041000000,actual_training_already_performed=False,estimates_are_not_guarantees=True,prior_export_scope='A800->.100 actual56-60s/520MB each; A100->.100 bandwidth not measured; reserve total600s, both stage max600 remain under same3600 clock',no_recipe_change=True)
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
