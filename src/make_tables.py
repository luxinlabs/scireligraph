#!/usr/bin/env python3
"""Emit LaTeX tables for the paper directly from results/ (no hand-copied numbers)."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / 'results'; O = ROOT / 'paper' / 'tables'; O.mkdir(parents=True, exist_ok=True)
S = json.loads((R / 'summary.json').read_text())
M = S['models']
BS = r'\\'


def write(name, lines):
    (O / f'{name}.tex').write_text('\n'.join(lines) + '\n')


# ---------------- Table: main results ----------------
ROWS = [('B0 Prevalence', 'B0', 'Training-fold prevalence', 'strict'),
        ('B1 Original-study metadata (LR)', 'B1', 'Original-study metadata, LR', 'strict'),
        ('B4 Metadata + flat citation evidence', 'B4', 'B1 + flat citation counts', 'strict'),
        ('M1 Metadata + evidence graph', 'M1', 'B4 + graph structure', 'strict'),
        ('M2 Metadata + graph + dependency correction', 'M2', 'M1 + dependency correction', 'strict'),
        ('D Metadata + replication design (launch-time)', 'D', 'B1 + planned replication design', 'launch'),
        ('Dx Metadata + achieved replication size (post-launch)', r'D$^{*}$', 'D + achieved size/power', 'post'),
        ('L-M2 Graph + dependency, NO cutoff (leaky)', r'M2$_\infty$', 'M2', 'none')]  # B1g, B2, NC, G-inf: artifact table
SHORT = {k: i for k, i, _, _ in ROWS} | {'G-only Graph + dependency (no metadata)': 'G', 'S-M2 Graph at replication-launch cutoff': r'M2$_\ell$', 'B1g Original-study metadata (GBT)': 'B1g', 'B2 Title/abstract TF-IDF (LR)': 'B2',
                                        'NC M2 with permuted graph (negative control)': 'NC',
                                        'LG-only Graph, NO cutoff (leaky)': r'G$_\infty$'}
b0 = M['B0 Prevalence']['mean']['brier']
L = [r"\begin{table*}[t]\centering\caption{Out-of-fold forecasting performance on 274 SCORE claims (164 papers). AUROC: mean$\pm$SD over 20 repetitions of paper-grouped 5-fold CV, with paper-clustered bootstrap 95\% CI of the seed-averaged predictions. Brier: mean over repetitions (lower is better). BSS: Brier skill score vs.\ B0. Cutoff ``post'': uses information available only after replication data collection (leakage illustration). Selected configurations; all others (GBT, text-only, graph-only, permuted, launch-cutoff M2) are in the artifact.}\label{tab:main}",
     r'\footnotesize\setlength{\tabcolsep}{6pt}\begin{tabular}{@{}llccccc@{}}\toprule',
     r'ID & Features & Cutoff & AUROC & 95\% CI & Brier & BSS ' + BS + r'\midrule']
for k, i, d, c in ROWS:
    m = M[k]; lo, hi = m['ci95']['auroc']
    if k.startswith(('D ', 'L-M2')): L.append(r'\midrule')
    L.append(f"{i} & {d} & {c} & {m['mean']['auroc']:.3f}$\\pm${m['sd']['auroc']:.3f} & [{lo:.2f}, {hi:.2f}] & "
             f"{m['mean']['brier']:.4f} & {100 * (1 - m['mean']['brier'] / b0):+.1f}\\% {BS}")
L += [r'\bottomrule\end{tabular}\end{table*}']
write('main', L)

# ---------------- Table: paired deltas (Holm) ----------------
D = pd.read_json(R / 'paired_deltas.json')
Q = {('B1', 'B0'): 'metadata signal', ('B4', 'B1'): 'flat evidence', ('M1', 'B4'): 'H1: graph vs.\\ flat',
     ('M2', 'M1'): 'H2: dependency correction', ('M2', 'B1'): 'evidence overall', ('M2', 'NC'): 'graph vs.\\ permuted',
     (r'M2$_\infty$', 'M2'): 'H3: no cutoff (M2)', (r'G$_\infty$', 'G'): 'H3: no cutoff (graph)',
     ('D', 'B1'): 'planned design', (r'D$^{*}$', 'D'): 'achieved size (leak)', ('B2', 'B0'): 'text signal'}
L = [r'\begin{table*}[t]\centering\caption{Paired differences on identical paper-clustered bootstrap resamples (2000; seed-averaged predictions). $p_{\mathrm{Holm}}$: two-sided bootstrap $p$ for $\Delta$AUROC, Holm-adjusted over the %d comparisons. $\Delta$Brier $<0$ favours the first model.}\label{tab:delta}' % len(D),
     r'\footnotesize\setlength{\tabcolsep}{6pt}\begin{tabular}{@{}lccccc@{}}\toprule',
     r'Comparison (question) & $\Delta$AUROC & 95\% CI & $p_{\mathrm{Holm}}$ & $\Delta$Brier & 95\% CI ' + BS + r'\midrule']
for _, r in D.iterrows():
    a, b = SHORT[r.a], SHORT[r.b]
    L.append(f"{a} vs.\\ {b} ({Q[(a, b)]}) & {r.d_auroc:+.3f} & [{r.d_auroc_ci[0]:+.3f}, {r.d_auroc_ci[1]:+.3f}] & {r.p_holm:.2f} & "
             f"{r.d_brier:+.4f} & [{r.d_brier_ci[0]:+.4f}, {r.d_brier_ci[1]:+.4f}] {BS}")
L += [r'\bottomrule\end{tabular}\end{table*}']
write('delta', L)

# ---------------- Table: selective prediction ----------------
Sel = pd.DataFrame(json.loads((R / 'selective.json').read_text()))
LAB = {'B1 Original-study metadata (LR)': 'B1', 'M2 Metadata + graph + dependency correction': 'M2', 'D Metadata + replication design (launch-time)': 'D'}
covs = sorted(Sel.coverage.unique())
L = [r"\begin{table}[t]\centering\caption{Selective prediction: accuracy when the agent acts only on the most confident fraction (coverage) of verifier outputs; paper-clustered 95\% CIs; confidence $|p-0.5|$. ``Pos'': share of the top 10\% that are \emph{replicate} calls.}\label{tab:selective}",
     r'\scriptsize\setlength{\tabcolsep}{2.2pt}\begin{tabular}{@{}l' + 'c' * len(covs) + r'c@{}}\toprule',
     'Model & ' + ' & '.join(f'{int(c * 100)}\\%' for c in covs) + ' & Pos ' + BS + r'\midrule']
for conf, title in [('abs05', None)]:  # robustness definition |p - median(p)| reported in the text only (space)
    for m in LAB:
        g = Sel[(Sel.model == m) & (Sel.confidence == conf)].sort_values('coverage')
        cells = [f"{a:.2f}\\,{{\\tiny[{c[0]:.2f},{c[1]:.2f}]}}" if cv < 1 else f"{a:.2f}" for a, c, cv in zip(g.accuracy, g.ci, g.coverage)]
        L.append(LAB[m] + ' & ' + ' & '.join(cells) + f" & {g.frac_pred_replicate.iloc[0]:.2f} {BS}")
L += [r'\bottomrule\end{tabular}\end{table}']
write('selective', L)

# ---------------- Table: criterion sensitivity ----------------
Sn = S['sensitivity']
NM = {'alt_analyst': 'Analyst judgement', 'alt_rep_wthn': 'Repl.\\ within orig.\\ CI', 'alt_bayes_rep': 'Bayesian replication'}
L = [r'\begin{table}[t]\centering\caption{Sensitivity to the replication-success criterion (AUROC, mean over 20 repetitions).}\label{tab:sens}',
     r'\scriptsize\begin{tabular}{@{}lcccc@{}}\toprule', r'Criterion & $n$ & Success rate & B1 & M2 ' + BS + r'\midrule']
for lab in NM:
    b1 = Sn[f'{lab} | B1 Original-study metadata (LR)']; m2 = Sn[f'{lab} | M2 Metadata + graph + dependency correction']
    L.append(f"{NM[lab]} & {b1['n']} & {b1['rate']:.3f} & {b1['auroc']:.3f} & {m2['auroc']:.3f} {BS}")
L += [r'\bottomrule\end{tabular}\end{table}']
write('sens', L)

# ---------------- Table: single-feature signal ----------------
F = pd.read_csv(R / 'feature_auroc.csv')
NAMES = {'a_neglog10_p': r'$-\log_{10} p$ (orig.)', 'a_power50': 'Orig.\\ power for 50\\% effect', 'a_p_exact': '$p$ reported exactly',
         'a_abs_r': r'$|r|$ (orig.)', 'a_log10_n': r'$\log_{10} N$ (orig.)', 'a_p_marginal': r'$0.01<p\le0.05$',
         'a_pub_year': 'Publication year', 'a_interaction': 'Interaction effect', 'g_cross_field_frac': 'Cross-field citer share',
         'g_log_refs': 'log references', 'g_venue_entropy': 'Citing-venue entropy', 'g_indep_ratio': 'Indep.\\ groups / citers',
         'g_log_indep_groups': 'log indep.\\ support groups', 'g_log_cites': 'log citations', 'g_selfcite_frac': 'Self-citation share',
         'g_coupling_mean': 'Mean coupling', 'g_internal_density': 'Citer--citer density', 'g_log_citing_authors': 'log citing authors',
         'g_field_count': 'Citing fields', 'g_log_cites_per_year': 'log citations/year', 'g_has_evidence': 'Any evidence',
         'd_log10_planned_n': r'$\log_{10} N$ planned (repl.)', 'd_new_data': 'New data (repl.)',
         'dx_log10_rep_n': r'$\log_{10} N$ achieved (repl.)', 'dx_power50': 'Achieved power (repl.)'}
TIER = {'a': 'A', 'g': 'G', 'd': 'D', 'dx': 'D$^{*}$'}
fmt = lambda v: f'{v:.1f}' if abs(v) >= 100 else (f'{v:.2f}' if abs(v) >= 1 else f'{v:.3f}')
st = F[F.cutoff == 'strict'].set_index('feature'); lk = F[F.cutoff == 'leaky'].set_index('feature')
L = [r'\begin{table}[t]\centering\caption{Single-feature signal on all 274 claims (descriptive, not cross-validated). AUROC $>0.5$: higher values go with replication. Means for replicated (R) vs.\ non-replicated (N) claims. Tier: A original study; G graph at strict cutoff; D planned replication design; D$^{*}$ post-launch; $\infty$: no cutoff.}\label{tab:feat}',
     r'\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}llcccc@{}}\toprule', r'Feature & Tier & AUROC & Mean R & Mean N & AUROC$_\infty$ ' + BS + r'\midrule']
for f, r in st.iterrows():
    inf = f"{lk.loc[f, 'auroc']:.3f}" if f in lk.index else '--'
    L.append(f"{NAMES.get(f, f.replace('_', ' '))} & {TIER[f.split('_')[0]]} & {r.auroc:.3f} & {fmt(r.mean_rep)} & {fmt(r.mean_nonrep)} & {inf} {BS}")
L += [r'\bottomrule\end{tabular}\end{table}']
write('feat', L)

# ---------------- Table: subgroups ----------------
Sb = pd.read_csv(R / 'subgroups.csv')
L = [r'\begin{table}[t]\centering\caption{Per-subgroup AUROC of seed-averaged out-of-fold predictions (descriptive; small $n$).}\label{tab:sub}',
     r'\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}lcccc@{}}\toprule', r'Subgroup & $n$ & Repl.\ rate & B1 & M2 ' + BS + r'\midrule']
for (g, v), d in Sb.groupby(['group', 'value'], sort=False):
    b1 = d[d.model.str.startswith('B1')].iloc[0]; m2 = d[d.model.str.startswith('M2')].iloc[0]
    L.append(f"{str(v).replace(' and ', ' \\& ').capitalize()} & {b1.n} & {b1.rate:.2f} & {b1.auroc:.3f} & {m2.auroc:.3f} {BS}")
L += [r'\bottomrule\end{tabular}\end{table}']
write('sub', L)

# ---------------- Table: contract tests x mutants ----------------
MU = json.loads((R / 'mutation.json').read_text())
TESTS = [('test_r1_future_evidence_is_invisible', 'R1', 'Metamorphic: citing works dated after \\texttt{as\\_of} added to every paper leave output unchanged'),
         ('test_r1_past_evidence_is_visible', 'R1', 'Positive control: works dated before \\texttt{as\\_of} are counted'),
         ('test_r1_refuses_on_or_after_launch', 'R1', 'Call on/after the replication launch is refused'),
         ('test_rejects_malformed_calls', '--', 'Malformed date or unknown stage is refused (2 cases)'),
         ('test_r3_own_paper_labels_never_used', 'R3', "Flipping all labels of the queried claim's paper leaves its prediction unchanged"),
         ('test_stage_gating_publication_ignores_launch_tier', 'R1', 'Publication stage: Tier-D values have no influence'),
         ('test_r4_output_is_decision_relevant', 'R4', 'Base rate equals training prevalence; weak-signal advice when selective test fails'),
         ('test_r5_every_call_is_audited', 'R5', 'Each call appends an audit record with provenance hashes')]
MUT = [('leaky_cutoff', 'leaky'), ('no_leave_paper_out', 'no-LPO'), ('no_stage_gating', 'no-gate'), ('no_refusal', 'no-refuse'), ('no_audit', 'no-audit')]
L = [r'\begin{table*}[t]\centering\caption{Contract tests for the verifier tool and mutation analysis. Each mutant removes one guarantee from the implementation; $\times$ marks the tests that fail (kill the mutant). The unmodified verifier passes all %s.}\label{tab:contract}' % MU['original']['summary'].split(' in ')[0].replace('passed', 'tests'),
     r'\footnotesize\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}lp{0.52\textwidth}' + 'c' * len(MUT) + r'@{}}\toprule',
     'Req. & Contract test & ' + ' & '.join(l for _, l in MUT) + ' ' + BS + r'\midrule']
for t, req, desc in TESTS:
    L.append(f"{req} & {desc} & " + ' & '.join('$\\times$' if t in MU[m]['failing_tests'] else '' for m, _ in MUT) + f' {BS}')
L += [r'\midrule', r'\multicolumn{2}{@{}l}{Mutant killed?} & ' + ' & '.join('yes' if MU[m]['killed'] else '\\textbf{no}' for m, _ in MUT) + f' {BS}',
      r'\bottomrule\end{tabular}\end{table*}']
write('contract', L)
print('tables written:', sorted(p.name for p in O.glob('*.tex')))

# ---------------- Table: LLM-as-judge (only if the LLM run has been scored) ----------------
LS = R / 'llm_summary.json'
if LS.exists():
    Lj = json.loads(LS.read_text())
    RC = json.loads((R / 'llm_recalibration.json').read_text()) if (R / 'llm_recalibration.json').exists() else {}
    MODEL_LAB = {'gpt-5.4-mini-2026-03-17': r'GPT-5.4 mini (2025-08)', 'gpt-6.1-sol': r'GPT-6.1 sol (2026-04)', 'claude-haiku-4-5-20251001': r'Claude Haiku 4.5 (2025-07)', 'claude-opus-5-5': r'Claude Opus 5.5 (2026-06)'}
    L = [r'\begin{table*}[t]\centering\caption{LLM-as-judge verifiers on the same claims. \emph{Identified}: title, citation, abstract and original statistics; \emph{de-identified}: field and original statistics only. AUROC with paper-clustered 95\% CI; $\Delta$ vs.\ B1 on identical claims and resamples. Probe: share answering that the paper was replicated in SCORE (\emph{aware}) for real SCORE papers vs.\ journal-matched decoys that were never replicated, and accuracy of named outcomes (counted whenever an outcome is named, regardless of stated awareness) vs.\ the majority-outcome baseline on the same claims. Brier (Platt): after Platt scaling fitted on training folds only (paper-grouped CV, 20 repetitions). B1 here is scored on its seed-averaged predictions, so its Brier differs slightly from the per-seed mean in Table~\ref{tab:main}.}\label{tab:llm}',
         r'\footnotesize\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}llccccccc@{}}\toprule',
         r'Model & Input & AUROC & 95\% CI & $\Delta$AUROC vs.\ B1 & 95\% CI & Brier & Brier (Platt) & Top-10\% acc. ' + BS + r'\midrule']
    for m, res in Lj.items():
        if 'forecast' not in res: continue
        for k, lab in [('id', 'identified'), ('deid', 'de-identified')]:
            f = res['forecast'][k]; dl = res[f'delta_{k}_vs_B1']; sl = res['selective_top10'][k]
            rc = RC.get(f"{m}|forecast_{k}", {}).get('platt', {})
            L.append(f"{MODEL_LAB.get(m, m) if k == 'id' else ''} & {lab} & {f['auroc']:.3f} & [{f['auroc_ci'][0]:.2f}, {f['auroc_ci'][1]:.2f}] & "
                     f"{dl['d_auroc']:+.3f} & [{dl['d_auroc_ci'][0]:+.3f}, {dl['d_auroc_ci'][1]:+.3f}] & {f['brier']:.4f} & "
                     f"{rc['brier']:.4f} & {sl['acc_top10']:.2f} {BS}" if rc else '')
    r0 = next(iter(r for r in Lj.values() if 'forecast' in r)); b1 = r0['forecast']['B1']
    L += [r'\midrule', f"B1 (metadata LR) & same claims & {b1['auroc']:.3f} & [{b1['auroc_ci'][0]:.2f}, {b1['auroc_ci'][1]:.2f}] & -- & -- & {b1['brier']:.4f} & -- & {r0['selective_top10']['B1']['acc_top10']:.2f} {BS}",
          r'\midrule', r'\multicolumn{9}{@{}l}{\emph{Memorization probe}} ' + BS,
          r'Model & & Aware (real) & Aware (decoy) & Named outcomes & Named acc. & Majority & & ' + BS + r'\midrule']
    for m, res in Lj.items():
        p = res['probe']
        acc = f"{p['named_outcome_accuracy']:.2f}" if p['named_outcome_accuracy'] is not None else '--'
        maj = f"{p['named_outcome_majority_baseline']:.2f}" if p.get('named_outcome_majority_baseline') is not None else '--'
        L.append(f"{MODEL_LAB.get(m, m)} & & {100 * p['aware_yes_real']:.1f}\\% & {100 * p['aware_yes_decoy']:.1f}\\% & {p['named_outcome_real']}/{p['n_real']} & {acc} & {maj} & & {BS}")
    L += [r'\bottomrule\end{tabular}\end{table*}']
    write('llm', L)
    print('llm table written')

# ---------------- Table: agent-in-the-loop ----------------
AGS = R / 'agent_summary.json'
if AGS.exists():
    ag = json.loads(AGS.read_text())
    ORDER = ['gpt-5.4-mini-2026-03-17', 'gpt-6.1-sol', 'claude-haiku-4-5-20251001', 'claude-opus-5-5']
    LAB = {'gpt-5.4-mini-2026-03-17': 'GPT-5.4 mini', 'gpt-6.1-sol': 'GPT-6.1 sol', 'claude-haiku-4-5-20251001': 'Claude Haiku 4.5', 'claude-opus-5-5': 'Claude Opus 5.5'}
    L = [r"\begin{table}[t]\centering\caption{Agent-in-the-loop triage on identical claims (scenario A: 274 valid queries; B: 259 post-launch queries, where the unprotected tool silently returns a no-cutoff prediction and the contract-aware tool refuses). U: agent sees only \textnormal{\texttt{p\_replicate}}; C: full contract output. Wrong auto = accepted non-replicating plus rejected replicating claims, as a share of all claims; $\Delta$ = C$-$U with paper-clustered 95\% CI. Revised wordings: Table~\ref{tab:r6}.}\label{tab:agent}",
         r'\scriptsize\setlength{\tabcolsep}{2pt}\begin{tabular}{@{}llccccc@{}}\toprule',
         r'& Agent & Esc.\ U & Esc.\ C & Wrong U & Wrong C & $\Delta$ wrong (pp) ' + BS + r'\midrule']
    for sc in 'AB':
        first = True
        for m in ORDER:
            if m not in ag: continue
            S = ag[m][sc]; u, c = S['unprotected'], S['contract_aware']; dd = S['diff_contract_minus_unprotected']['auto_error']
            L.append(f"{sc if first else ''} & {LAB[m]} & {100 * u['escalate']:.0f}\\% & {100 * c['escalate']:.0f}\\% & {100 * u['auto_error']:.1f}\\% & "
                     f"{100 * c['auto_error']:.1f}\\% & {100 * dd['point']:+.1f} [{100 * dd['ci'][0]:+.1f}, {100 * dd['ci'][1]:+.1f}] {BS}")
            first = False
            if False and 'contract_aware_v2' in S:  # revised wordings now in Table r6
                v = S['contract_aware_v2']
                L.append(f" & {LAB[m]}, C$^{{\\dagger}}$ & & {100 * v['escalate']:.0f}\\% & & {100 * v['auto_error']:.1f}\\% & {BS}")
        if sc == 'A': L.append(r'\midrule')
    L += [r'\bottomrule\end{tabular}\end{table}']
    write('agent', L)
    print('agent table written')

# ---------------- Table: agent control arms (scenario A) ----------------
if AGS.exists():
    ag = json.loads(AGS.read_text())
    if all('no_tool' in ag[m]['A'] and 'placebo' in ag[m]['A'] for m in ORDER if m in ag):
        L = [r"\begin{table}[t]\centering\caption{Control arms for scenario~A (274 claims): share escalated / share of wrong automatic decisions. \emph{Unavailable}: the agent sees the claim and a note that the tool output is unavailable (a cue in itself). \emph{Placebo}: \textnormal{\texttt{p\_replicate}} plus a generic ``this estimate is uncertain'' note. \emph{Contract}: full \textnormal{\texttt{verify\_claim}} output. \emph{Always escalate} needs no model.}\label{tab:controls}",
             r'\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}lcccc@{}}\toprule',
             r'Agent & Unavailable & Placebo & Unprotected & Contract ' + BS + r'\midrule']
        f = lambda x: f"{100 * x['escalate']:.0f}\\% / {100 * x['auto_error']:.1f}\\%"
        for m in ORDER:
            if m not in ag: continue
            A_ = ag[m]['A']
            L.append(f"{LAB[m]} & {f(A_['no_tool'])} & {f(A_['placebo'])} & {f(A_['unprotected'])} & {f(A_['contract_aware'])} {BS}")
        ae = next(iter(ag.values()))['A']['always_escalate']
        L += [r'\midrule', f"Always escalate & \\multicolumn{{4}}{{c}}{{{f(ae)}}} {BS}", r'\bottomrule\end{tabular}\end{table}']
        write('controls', L)
        print('controls table written')
