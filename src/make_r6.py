#!/usr/bin/env python3
"""Table, macros and figure for the pre-specified R6 validation, the budget analysis and the refit bootstrap.
Writes paper/tables/r6.tex, paper/tables/r6_numbers.tex and paper/figures/risk_coverage.pdf."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; T = ROOT / 'paper' / 'tables'; FIG = ROOT / 'paper' / 'figures'
R6 = json.loads((ROOT / 'results/r6_summary.json').read_text())
AG = json.loads((ROOT / 'results/agent_summary.json').read_text())
BU = json.loads((ROOT / 'results/budget_analysis.json').read_text())
RF = json.loads((ROOT / 'results/refit_bootstrap.json').read_text())
MODELS = [('gpt-5.4-mini-2026-03-17', 'GPT-5.4 mini', 'Mini'), ('gpt-6.1-sol', 'GPT-6.1 sol', 'Sol'),
          ('claude-haiku-4-5-20251001', 'Claude Haiku 4.5', 'Haiku'), ('claude-opus-5-5', 'Claude Opus 5.5', 'Opus')]
pct = lambda x, k=0: f'{100 * x:.{k}f}\\%'
N = {}


def put(k, v): N[k] = v


rows = []
r6cells_esc, r6cells_wrong = [], []
for m, name, tag in MODELS:
    A = R6[m]['A']; S = R6[m]['S']
    cells = [w for w in A if w[:2] in ('v2', 'v3', 'v4')]
    esc = min(A[w]['escalate'] for w in cells); wrong = max(A[w]['wrong'] for w in cells)
    r6cells_esc.append(esc); r6cells_wrong.append(wrong)
    orig = AG[m]['A']['contract_aware']['auto_error']
    rows.append(f"{name.replace('Claude ', '')} & {pct(orig, 1)} & {pct(esc)}/{pct(wrong, 1)} & "
                f"{pct(S['bare']['auto'])}/{pct(S['bare']['wrong'], 1)} & {pct(S['v3']['auto'])}/{pct(S['v3']['wrong'], 1)} & {pct(S['v3']['agrees_with_license'])} \\\\")
    put(f'rs{tag}BareWrong', pct(S['bare']['wrong'], 1)); put(f'rs{tag}VWrong', pct(S['v3']['wrong'], 1))
    put(f'rs{tag}BareOut', pct(S['bare']['auto_outside_licensed'])); put(f'rs{tag}VOut', pct(S['v3']['auto_outside_licensed']))
    put(f'rs{tag}VIn', pct(S['v3']['auto_in_licensed'])); put(f'rs{tag}Agree', pct(S['v3']['agrees_with_license']))
    C = R6[m]['comprehension']
    put(f'cq{tag}SelRep', pct(1 - C['CA_v1']['q3'])); put(f'cq{tag}SelRepVfour', pct(1 - C['CA_v4']['q3']))

L = [r"\begin{table}[t]\centering\caption{Pre-specified R6 validation (scenario A, 274 claims; wrong = wrong automatic decisions). "
     r"\emph{R6}: wordings v2--v4 under two system prompts (Opus: one), worst cell shown. "
     r"\emph{Decision-ready}: a synthetic verifier (AUROC " + f"{BU['synthetic_tool']['auroc']:.2f}" + r") whose v3 output licenses accept/reject for confident scores; "
     r"\emph{agree}: decisions consistent with the licensed actions.}\label{tab:r6}",
     r"\footnotesize\setlength{\tabcolsep}{2.5pt}\begin{tabular}{@{}lccccc@{}}\toprule",
     r" & Orig. & R6 & \multicolumn{3}{c}{Decision-ready tool} \\ \cmidrule(l){4-6}",
     r"Agent & wrong & esc./wrong & bare: auto/wrong & R6: auto/wrong & agree \\ \midrule", *rows, r"\bottomrule\end{tabular}\end{table}"]
(T / 'r6.tex').write_text('\n'.join(L) + '\n')

# global macros
st = BU['synthetic_tool']
put('rsAuroc', f"{st['auroc']:.2f}"); put('rsTailShare', pct(st['tail_share'])); put('rsTailAcc', pct(st['tail_acc']))
put('rsMinEsc', pct(min(r6cells_esc))); put('rsMaxWrong', pct(max(r6cells_wrong), 1))
put('rsCells', str(sum(len([w for w in R6[m]['A'] if w[:2] in ('v2', 'v3', 'v4')]) for m, _, _ in MODELS)))
put('rsCost', f"{sum(R6[m]['cost_usd'] for m, _, _ in MODELS):.2f}"); put('rsCalls', f"{sum(R6[m]['n_calls'] for m, _, _ in MODELS):,}".replace(',', '{,}'))
hk = R6['claude-haiku-4-5-20251001']['A']
put('rsHaikuDvOne', f"{100 * hk['v3']['d_vs_v1']:+.1f}".replace('-', '\\mbox{$-$}')); put('rsHaikuDvOneCi', '[' + ', '.join(f"{100 * x:+.1f}" for x in hk['v3']['d_vs_v1_ci']).replace('-', '$-$') + ']')
Bsc = [R6[m]['B'][w]['escalate'] for m, _, _ in MODELS for w in ('v3', 'v4') if w in R6[m].get('B', {})]
put('rsBEscMin', pct(min(Bsc)))
comp = [R6[m]['comprehension'] for m, _, _ in MODELS]
put('cqVoneActMin', pct(min(min(c['CA_v1']['q2'], c['CA_v1']['q4']) for c in comp)))
put('cqVoneRefMin', pct(min(c['CB_v1']['r1'] for c in comp)))
put('cqReadyQtwo', pct(np.mean([c['CS_v3']['q2_per_claim_key'] for c in comp])))
put('cqReadyQtwoFrozen', pct(np.mean([c['CS_v3']['q2'] for c in comp]))); put('cqReadyNlic', str(comp[0]['CS_v3']['n_licensed'])); put('cqReadyN', str(comp[0]['CS_v3']['n']))
for e in ('0.25', '0.5', '0.75'):
    k = {'0.25': 'TwentyFive', '0.5': 'Fifty', '0.75': 'SeventyFive'}[e]
    put(f'buWeak{k}', pct(BU['tools']['weak']['budgets'][e]['wrong_share_of_all'], 1))
    put(f'buStrong{k}', pct(BU['tools']['strong']['budgets'][e]['wrong_share_of_all'], 1))
    put(f'buWeakRisk{k}', pct(BU['tools']['weak']['budgets'][e]['risk']))
put('rfBoneCi', f"[{RF['B1_auroc_ci'][0]:.2f}, {RF['B1_auroc_ci'][1]:.2f}]"); put('rfBoneGt', pct(RF['B1_auroc_gt_05_share']))
put('rfMtwoBoneCi', f"[{RF['M2_minus_B1_auroc_ci'][0]:+.3f}, {RF['M2_minus_B1_auroc_ci'][1]:+.3f}]".replace('-', '$-$'))
(T / 'r6_numbers.tex').write_text(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in N.items()))

# figure: risk-coverage with agent operating points
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, ax = plt.subplots(figsize=(3.4, 2.15))
g = np.array(BU['grid'])
for name, col, lab in (('weak', '#b2182b', 'real verifier (tool output)'), ('strong', '#2166ac', 'decision-ready (synthetic)')):
    t = BU['tools'][name]
    ax.plot(g, t['risk'], color=col, lw=1.2, label=lab); ax.fill_between(g, t['lo'], t['hi'], color=col, alpha=0.15, lw=0)
ax.axhline(BU['base_error'], color='grey', ls=':', lw=0.8); ax.text(0.50, BU['base_error'] + 0.012, 'majority-class error', color='grey', fontsize=6)
mk = {'unprotected': ('o', 'bare number'), 'placebo': ('s', 'placebo'), 'contract_aware': ('^', 'contract (orig.)'),
      'bare': ('o', None), 'v3': ('D', 'R6 contract (v3)')}
seen = set()
for p in BU['agent_points']:
    if p['risk'] is None or p['arm'] not in mk: continue
    m_, lab = mk[p['arm']]; col = '#b2182b' if p['tool'] == 'weak' else '#2166ac'
    lab = lab if lab and lab not in seen else None; seen.add(lab)
    ax.scatter(p['coverage'], p['risk'], marker=m_, s=12, facecolors='none' if p['tool'] == 'weak' else col, edgecolors=col, lw=0.7,
               label=lab, zorder=3)
ax.set_xlabel('coverage (share of claims decided automatically)'); ax.set_ylabel('error among automated')
ax.set_xlim(0, 1.0); ax.set_ylim(0, 0.62); ax.legend(fontsize=5.5, ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handletextpad=0.3)
ax.spines[['top', 'right']].set_visible(False); fig.tight_layout(pad=0.2)
FIG.mkdir(exist_ok=True); fig.savefig(FIG / 'risk_coverage.pdf')
print(json.dumps(N, indent=0))
