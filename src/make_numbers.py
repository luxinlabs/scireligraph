#!/usr/bin/env python3
"""Every number quoted in the paper's prose is emitted here as a LaTeX macro, straight from results/ and data/.
The manuscript must not contain hand-typed result numbers."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'; P = ROOT / 'data/processed'
OUT = ROOT / 'paper' / 'tables'
S = json.loads((R / 'summary.json').read_text()); M = S['models']
man = json.loads((P / 'manifest.json').read_text())
D = pd.read_json(R / 'paired_deltas.json'); sel = pd.DataFrame(json.loads((R / 'selective.json').read_text()))
feat = pd.read_csv(R / 'feature_auroc.csv'); sub = pd.read_csv(R / 'subgroups.csv')
perm = json.loads((R / 'permutation_null.json').read_text())
mut = json.loads((R / 'mutation.json').read_text())
starter = json.loads((R / 'starter_baseline/summary.json').read_text())['models']
claims = pd.read_csv(P / 'claims.csv')
N = {}


def put(name, value):
    assert name.isalpha(), name
    N[name] = value


def m(key): return M[key]
def d(a, b): return D[(D.a.str.startswith(a)) & (D.b.str.startswith(b))].iloc[0]
def f3(x): return f'{x:.3f}'
def f2(x): return f'{x:.2f}'
def sgn(x, k=3):
    v = f'{x:+.{k}f}'  # unbreakable box with a true minus/plus so values never split across lines
    return '\\mbox{$' + v[0] + '$' + v[1:] + '}'
def ci(lo, hi, k=3): return f'[{sgn(lo, k)}, {sgn(hi, k)}]'
def pct(x): return f'{100 * x:.0f}\\%'


B0b = m('B0 Prevalence')['mean']['brier']
for tag, key in [('Bone', 'B1 Original-study metadata (LR)'), ('Mtwo', 'M2 Metadata + graph + dependency correction'),
                 ('Dplan', 'D Metadata + replication design (launch-time)'), ('Dach', 'Dx Metadata + achieved replication size (post-launch)'),
                 ('Gonly', 'G-only Graph + dependency (no metadata)'), ('Btwo', 'B2 Title/abstract TF-IDF (LR)'), ('Bgbt', 'B1g Original-study metadata (GBT)'),
                 ('Mtwolaunch', 'S-M2 Graph at replication-launch cutoff'), ('Bzero', 'B0 Prevalence')]:
    x = m(key)
    put(f'auc{tag}', f3(x['mean']['auroc'])); put(f'sd{tag}', f3(x['sd']['auroc']))
    put(f'ci{tag}', f"[{x['ci95']['auroc'][0]:.2f}, {x['ci95']['auroc'][1]:.2f}]")
    put(f'brier{tag}', f"{x['mean']['brier']:.4f}"); put(f'bss{tag}', f"{100 * (1 - x['mean']['brier'] / B0b):.1f}\\%")
for tag, a, b in [('BoneBzero', 'B1', 'B0'), ('BfourBone', 'B4', 'B1'), ('MoneBfour', 'M1', 'B4'), ('MtwoMone', 'M2', 'M1'),
                  ('MtwoBone', 'M2', 'B1'), ('MtwoNC', 'M2', 'NC'), ('LeakMtwo', 'L-M2', 'M2'), ('LeakG', 'LG-only', 'G-only'),
                  ('DplanBone', 'D ', 'B1'), ('DachDplan', 'Dx', 'D ')]:
    r = d(a, b)
    put(f'd{tag}', sgn(r.d_auroc)); put(f'dci{tag}', ci(*r.d_auroc_ci)); put(f'holm{tag}', f2(r.p_holm))
    put(f'dbrier{tag}', sgn(r.d_brier, 4)); put(f'dbrierci{tag}', ci(*r.d_brier_ci, 4))
put('punBoneBzero', f2(d('B1', 'B0').p_two_sided))

# selective (primary |p-0.5|)
for tag, key in [('Bone', 'B1 Original-study metadata (LR)'), ('Mtwo', 'M2 Metadata + graph + dependency correction'), ('Dplan', 'D Metadata + replication design (launch-time)')]:
    g = sel[(sel.model == key) & (sel.confidence == 'abs05')].set_index('coverage')
    put(f'selTop{tag}', f2(g.loc[0.1, 'accuracy'])); put(f'selTopCI{tag}', f"[{g.loc[0.1, 'ci'][0]:.2f}, {g.loc[0.1, 'ci'][1]:.2f}]")
    put(f'selHalf{tag}', f2(g.loc[0.5, 'accuracy'])); put(f'selAll{tag}', f2(g.loc[1.0, 'accuracy']))
    put(f'selPos{tag}', pct(g.loc[0.1, 'frac_pred_replicate']))
put('selPosMin', pct(sel[(sel.confidence == 'abs05') & (sel.coverage == 0.1)].frac_pred_replicate.min()))
put('selPosMax', pct(sel[(sel.confidence == 'abs05') & (sel.coverage == 0.1)].frac_pred_replicate.max()))

# permutation null
put('permK', str(perm['K'])); put('permSeeds', str(perm['seeds'])); put('permReal', f3(perm['real_auroc']))
put('permNullMean', f3(perm['null_auroc_mean'])); put('permNullLo', f3(perm['null_auroc_q'][0])); put('permNullHi', f3(perm['null_auroc_q'][2]))
put('permP', f2(perm['p_auroc'])); put('permBone', f3(perm['b1_auroc'])); put('permPbrier', f2(perm['p_brier']))

# data / harness facts
put('nCiting', f"{man['openalex_coverage']['citing_works']:,}".replace(',', '{,}'))
put('nCitingStrict', f"{man['strict_citing_works']:,}".replace(',', '{,}'))
put('earliestStart', man['earliest_replication_start']); put('nLaunch', str(man['launch_cutoff_available']))
put('nPlanned', str(int(claims.d_log10_planned_n.notna().sum())))
put('plannedRatio', f"{man['planned_vs_achieved_n_median_ratio']:.2f}")
a = S['graph_audit']
put('logCitesStrict', f2(a['strict']['g_log_cites'])); put('logCitesLeaky', f2(a['leaky']['g_log_cites']))
put('hasEvStrict', f"{100 * a['strict']['g_has_evidence']:.1f}\\%"); put('hasEvLeaky', f"{100 * a['leaky']['g_has_evidence']:.0f}\\%")
fs = feat[feat.cutoff == 'strict'].set_index('feature')
put('fpRep', f"{fs.loc['a_neglog10_p', 'mean_rep']:.1f}"); put('fpNon', f"{fs.loc['a_neglog10_p', 'mean_nonrep']:.1f}"); put('fpAuc', f2(fs.loc['a_neglog10_p', 'auroc']))
put('fpowRep', f2(fs.loc['a_power50', 'mean_rep'])); put('fpowNon', f2(fs.loc['a_power50', 'mean_nonrep'])); put('fpowAuc', f2(fs.loc['a_power50', 'auroc']))
put('fmargRep', pct(fs.loc['a_p_marginal', 'mean_rep'])); put('fmargNon', pct(fs.loc['a_p_marginal', 'mean_nonrep']))
put('fexactRep', pct(fs.loc['a_p_exact', 'mean_rep'])); put('fexactNon', pct(fs.loc['a_p_exact', 'mean_nonrep']))
put('fxfRep', f2(fs.loc['g_cross_field_frac', 'mean_rep'])); put('fxfNon', f2(fs.loc['g_cross_field_frac', 'mean_nonrep']))
put('fxfAuc', f2(1 - fs.loc['g_cross_field_frac', 'auroc'])); put('fcitesAuc', f2(fs.loc['g_log_cites', 'auroc']))
sb = sub[sub.model.str.startswith('B1')].set_index('value')
put('subPsych', f2(sb.loc['psychology and health', 'auroc'])); put('subPsychN', str(sb.loc['psychology and health', 'n']))
put('subEcon', f2(sb.loc['economics and finance', 'auroc'])); put('subEconN', str(sb.loc['economics and finance', 'n']))
put('subPol', f2(sb.loc['political science', 'auroc'])); put('subPolN', str(sb.loc['political science', 'n']))
put('starterAuc', f3(starter['metadata_logistic']['auroc'])); put('starterBrier', f3(starter['metadata_logistic']['brier']))
put('starterPrevBrier', f3(starter['prevalence']['brier']))
sens = S['sensitivity']
b1s = [v['auroc'] for k, v in sens.items() if 'B1 ' in k]
put('sensLo', f2(min(b1s))); put('sensHi', f2(max(b1s)))
put('nTests', mut['original']['summary'].split(' passed')[0].strip())
put('nMutants', str(len(mut) - 1)); put('nKilled', str(sum(1 for k, v in mut.items() if k != 'original' and v['killed'])))
put('locPython', str(sum(len(p.read_text().splitlines()) for p in (ROOT / 'src').glob('*.py'))))

pw = S['paper_weighted']
put('pwBone', f3(pw['B1 Original-study metadata (LR)']['auroc'])); put('pwMtwo', f3(pw['M2 Metadata + graph + dependency correction']['auroc']))
put('pwDplan', f3(pw['D Metadata + replication design (launch-time)']['auroc'])); put('pwBase', f"{100 * pw['B0 Prevalence']['base_rate']:.1f}\\%")
put('pwBssBone', f"{100 * (1 - pw['B1 Original-study metadata (LR)']['brier'] / pw['B0 Prevalence']['brier']):.1f}\\%")
put('maxClaimsPaper', str(int(claims.paper_id.value_counts().max())))

LS = R / 'llm_summary.json'
if LS.exists():
    Lj = json.loads(LS.read_text())
    for tag, m in [('Mini', 'gpt-5.4-mini-2026-03-17'), ('Sol', 'gpt-6.1-sol'), ('Haiku', 'claude-haiku-4-5-20251001'), ('Opus', 'claude-opus-5-5')]:
        if m not in Lj: continue
        res = Lj[m]
        for k, kt in [('id', 'Id'), ('deid', 'Deid')]:
            f = res['forecast'][k]; dl = res[f'delta_{k}_vs_B1']
            put(f'llm{tag}{kt}Auc', f3(f['auroc'])); put(f'llm{tag}{kt}Ci', f"[{f['auroc_ci'][0]:.2f}, {f['auroc_ci'][1]:.2f}]")
            put(f'llm{tag}{kt}Brier', f"{f['brier']:.4f}")
            put(f'llm{tag}{kt}VsBone', sgn(dl['d_auroc'])); put(f'llm{tag}{kt}VsBoneCi', ci(*dl['d_auroc_ci']))
            put(f'llm{tag}{kt}Top', f2(res['selective_top10'][k]['acc_top10'])); put(f'llm{tag}{kt}All', f2(res['selective_top10'][k]['acc_all']))
            put(f'llm{tag}{kt}Sd', f2(res['selective_top10'][k]['pred_spread_sd']))
            sk = res['selective_top10'][k]
            put(f'llm{tag}{kt}TopCi', f"[{sk['acc_top10_ci'][0]:.2f}, {sk['acc_top10_ci'][1]:.2f}]")
            put(f'llm{tag}{kt}GainCi', ci(*sk['gain_top10_minus_all_ci'], 2))
            put(f'llm{tag}{kt}TopPos', pct(sk['frac_pos_top10']))
        dd = res['delta_id_vs_deid']
        put(f'llm{tag}IdDeid', sgn(dd['d_auroc'])); put(f'llm{tag}IdDeidCi', ci(*dd['d_auroc_ci']))
        pr = res['probe']
        put(f'llm{tag}AwareReal', f"{100 * pr['aware_yes_real']:.1f}\\%"); put(f'llm{tag}AwareDecoy', f"{100 * pr['aware_yes_decoy']:.1f}\\%")
        put(f'llm{tag}Named', str(pr['named_outcome_real']))
        put(f'llm{tag}AwareRealCi', f"[{100 * pr['aware_yes_real_ci'][0]:.0f}, {100 * pr['aware_yes_real_ci'][1]:.0f}]\\%")
        put(f'llm{tag}AwareDecoyCi', f"[{100 * pr['aware_yes_decoy_ci'][0]:.0f}, {100 * pr['aware_yes_decoy_ci'][1]:.0f}]\\%")
        put(f'llm{tag}NamedDecoy', str(pr['named_outcome_decoy']))
        put(f'llm{tag}NamedAcc', f2(pr['named_outcome_accuracy']) if pr['named_outcome_accuracy'] is not None else '--')
        put(f'llm{tag}NamedMaj', f2(pr['named_outcome_majority_baseline']) if pr.get('named_outcome_majority_baseline') is not None else '--')
        put(f'llm{tag}Cost', f"{res['cost_usd']:.2f}"); put(f'llm{tag}Fail', str(res['parse_or_api_failures']))
        put(f'llm{tag}Effort', str((res.get('params') or {}).get('reasoning_effort', 'default')))
    put('llmBoneAuc', f3(Lj['gpt-6.1-sol']['forecast']['B1']['auroc'])); put('llmN', str(Lj['gpt-6.1-sol']['n_forecast']))
    put('llmBoneBrier', f"{Lj['gpt-6.1-sol']['forecast']['B1']['brier']:.4f}"); put('llmBzeroBrier', f"{Lj['gpt-6.1-sol']['forecast']['B0']['brier']:.4f}")
    put('llmBoneTop', f2(Lj['gpt-6.1-sol']['selective_top10']['B1']['acc_top10']))
    put('llmCalls', str(sum(v['calls_logged'] for v in Lj.values())))
    put('llmCostTotal', f"{sum(v['cost_usd'] for v in Lj.values()):.2f}")

RCF = R / 'llm_recalibration.json'
if RCF.exists():
    RC = json.loads(RCF.read_text())
    for tag, m in [('Mini', 'gpt-5.4-mini-2026-03-17'), ('Sol', 'gpt-6.1-sol'), ('Haiku', 'claude-haiku-4-5-20251001'), ('Opus', 'claude-opus-5-5')]:
        if f'{m}|forecast_id' not in RC: continue
        e = RC[f'{m}|forecast_id']
        for k, kt in [('platt', 'Platt'), ('isotonic', 'Iso')]:
            put(f'rc{tag}{kt}Brier', f"{e[k]['brier']:.4f}"); put(f'rc{tag}{kt}Auc', f3(e[k]['auroc']))
            put(f'rc{tag}{kt}Bss', f"{100 * e[k]['bss_vs_B0']:+.1f}\\%")
            put(f'rc{tag}{kt}VsBone', sgn(e[k]['d_brier_vs_B1'], 4)); put(f'rc{tag}{kt}VsBoneCi', ci(*e[k]['d_brier_vs_B1_ci'], 4))
            put(f'rc{tag}{kt}VsBzeroCi', ci(*e[k]['d_brier_vs_B0_ci'], 4))
MS = R / 'mutmut_summary.json'
if MS.exists():
    ms = json.loads(MS.read_text())
    put('mutTotal', str(ms['total'])); put('mutToolKilled', str(ms['tool_raw']['killed'])); put('mutToolScore', pct(ms['tool_raw']['score']))
    put('mutVoneKilled', str(ms['verified_v1']['killed'])); put('mutVoneScore', pct(ms['verified_v1']['score']))
    put('mutMissed', str(ms['verified_v1']['tool_missed_kills']))
    if ms.get('verified_v2'):
        v2 = ms['verified_v2']
        put('mutVtwoKilled', str(v2['killed'])); put('mutVtwoScore', pct(v2['score'])); put('mutVtwoTimeout', str(v2['timeout']))
        put('mutScore', pct(v2['score']))
    put('mutSurv', str(ms['remaining_survivors']))
    if ms.get('verified_v3'):
        put('mutVthreeKilled', str(ms['verified_v3']['killed'])); put('mutVthreeScore', pct(ms['verified_v3']['score']))
        put('mutVthreeExEq', pct(ms['verified_v3']['score_excluding_equivalent']))
    if ms.get('verified_v2') and ms['verified_v2'].get('score_excluding_equivalent'):
        put('mutScoreExEq', pct(ms['verified_v2']['score_excluding_equivalent']))
    for c, n in ms['survivor_classes'].items():
        put('mutCls' + ''.join(w.capitalize() for w in c.split('_')), str(n))
AS = R / 'agent_summary.json'
if AS.exists():
    ag = json.loads(AS.read_text())
    for tag, m in [('Mini', 'gpt-5.4-mini-2026-03-17'), ('Sol', 'gpt-6.1-sol'), ('Haiku', 'claude-haiku-4-5-20251001'), ('Opus', 'claude-opus-5-5')]:
        if m not in ag: continue
        for sc in 'AB':
            S = ag[m][sc]
            for wtag, w in [('U', 'unprotected'), ('C', 'contract_aware')]:
                for k in ('escalate', 'auto', 'unsafe_accept', 'wrong_reject', 'auto_error'):
                    put(f'ag{tag}{sc}{wtag}{k.replace("_", "").capitalize()}', pct(S[w][k]))
                put(f'ag{tag}{sc}{wtag}Autoacc', f2(S[w]['auto_accuracy']) if S[w]['auto_accuracy'] is not None else '--')
            for k in ('escalate', 'unsafe_accept', 'auto_error'):
                dd = S['diff_contract_minus_unprotected'][k]
                put(f'ag{tag}{sc}D{k.replace("_", "").capitalize()}', sgn(100 * dd['point'], 1))
                put(f'ag{tag}{sc}D{k.replace("_", "").capitalize()}Ci', ci(100 * dd['ci'][0], 100 * dd['ci'][1], 0))
            put(f'ag{tag}{sc}N', str(S['unprotected']['n']))
            # break-even reviewer cost c* (cost of one escalation relative to one wrong automatic decision):
            # utility = -(wrong auto) - c * (escalations); the contract is better iff c < -dErr/dEsc (when dEsc > 0)
            dE, dS = S['diff_contract_minus_unprotected']['auto_error']['point'], S['diff_contract_minus_unprotected']['escalate']['point']
            if dS > 0 and dE < 0:
                put(f'ag{tag}{sc}Breakeven', f2(-dE / dS))
            if 'contract_aware_v2' in S:
                put(f'ag{tag}{sc}Vtwo' + 'Esc', pct(S['contract_aware_v2']['escalate'])); put(f'ag{tag}{sc}Vtwo' + 'Err', pct(S['contract_aware_v2']['auto_error']))
            put(f'ag{tag}{sc}CRej', pct(S['contract_aware']['reject']))
            for key, kt in (('diff_contract_aware_minus_placebo', 'ConVsPla'), ('diff_unprotected_minus_no_tool', 'UnpVsNo'),
                            ('diff_contract_aware_minus_no_tool', 'ConVsNo')):
                if key in S:
                    put(f'ag{tag}{sc}{kt}', sgn(100 * S[key]['point'], 1)); put(f'ag{tag}{sc}{kt}Ci', ci(100 * S[key]['ci'][0], 100 * S[key]['ci'][1], 1))
            for w, wt in (('unprotected_p2', 'UPtwo'), ('placebo_p2', 'PlaPtwo'), ('contract_aware_p2', 'CPtwo')):
                if w in S:
                    put(f'ag{tag}{sc}{wt}Err', f"{100 * S[w]['auto_error']:.1f}\\%"); put(f'ag{tag}{sc}{wt}Rej', pct(S[w]['reject']))
            for w, wt in (('no_tool', 'Notool'), ('placebo', 'Placebo')):
                if w in S:
                    put(f'ag{tag}{sc}{wt}Esc', pct(S[w]['escalate'])); put(f'ag{tag}{sc}{wt}Err', f"{100 * S[w]['auto_error']:.1f}\\%")
                    put(f'ag{tag}{sc}{wt}Rej', pct(S[w]['reject']))
    scen = [json.loads(l) for l in (R / 'agent' / 'scenarios.jsonl').read_text().splitlines() if l.strip()]
    pa = [r['unprotected']['p_replicate'] for r in scen if r['scenario'] == 'A']
    put('agPmin', f2(min(pa))); put('agPmax', f2(max(pa))); put('agCalls', f"{2 * len(scen) * len(ag):,}".replace(',', '{,}')); put('agCallsVtwo', f"{sum(v['n_calls'] for v in ag.values()) - 2 * len(scen) * len(ag):,}".replace(',', '{,}'))
    put('agCost', f"{sum(v['cost_usd'] for v in ag.values()):.2f}")

(OUT / 'numbers.tex').write_text('\n'.join(f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in N.items()) + '\n')

# real tool output for the listing (no illustrative values)
smoke = json.loads((R / 'mcp_smoke.json').read_text())
pub = smoke['publication@2018-12-31']
show = {k: pub[k] for k in ['claim_id', 'stage', 'as_of', 'p_replicate', 'base_rate', 'cv_auroc', 'cv_brier_skill', 'selective_ok',
                            'n_evidence', 'tiers_used', 'n_train_claims']}
show['drivers'] = pub['drivers'][:1]
show['provenance'] = {'archive_sha256': pub['provenance']['archive_sha256'][:12] + '...', 'model_config': pub['provenance']['model_config']}
show['advice'] = pub['advice']
refused = smoke['publication@2023-01-01']
def compact(o, per=3):
    items = [f'"{k}": {json.dumps(v)}' for k, v in o.items()]
    rows = [', '.join(items[i:i + per]) for i in range(0, len(items), per)]
    return '{' + ',\n '.join(rows) + '}'
(OUT / 'tool_output.txt').write_text('// verify_claim("Ej3y_single-trace", as_of="2018-12-31")\n' + compact(show)
                                     + '\n// same claim, as_of="2023-01-01"\n' + compact(refused) + '\n')
print(len(N), 'macros written')
