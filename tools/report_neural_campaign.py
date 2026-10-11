"""Standalone public research report with direct measurements and negative findings."""
from pathlib import Path
import json,sys,html,statistics
report=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
rows=[]
for mode,values in report['distributions'].items():
    a=values['accuracy_values'];pairs=values['pair_accuracy_values']
    rows.append((mode,len(a),statistics.mean(a) if a else 0,min(a) if a else 0,max(a) if a else 0,statistics.mean(pairs) if pairs else 0))
body=''.join(f'<tr><td>{html.escape(mode)}</td><td>{n}</td><td>{mean:.2%}</td><td>{low:.2%}–{high:.2%}</td><td>{pair:.2%}</td></tr>' for mode,n,mean,low,high,pair in rows)
colors=['#6b7280','#14b8a6','#3b82f6','#8b5cf6'];bars=[]
for i,(mode,n,mean,low,high,pair) in enumerate(rows):
    y=70+i*75;bars.append(f'<text x="30" y="{y+20}" font-family="sans-serif" font-size="17">{mode}</text><rect x="160" y="{y}" width="{mean*500}" height="30" fill="{colors[i]}"/><text x="{175+mean*500}" y="{y+22}" font-family="sans-serif" font-size="17">{mean:.1%}</text>')
svg='<svg xmlns="http://www.w3.org/2000/svg" width="820" height="440"><rect width="820" height="440" fill="white"/><text x="30" y="35" font-family="sans-serif" font-size="23">Public development accuracy · five seeds</text>'+''.join(bars)+'<text x="30" y="405" font-family="sans-serif" font-size="15">Correlated synthetic diagnostics; independent generalization NOT ACHIEVED.</text></svg>'
(out/'SAAL-RSI-v0.9.3-Comparison.svg').write_text(svg)
page='<!doctype html><html><meta charset="utf-8"><title>SAAL-RSI v0.9.3 Research Results</title><style>body{max-width:1000px;margin:40px auto;font:18px system-ui;color:#173042}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:12px;border-bottom:1px solid #ccc}</style><h1>SAAL-RSI v0.9.3</h1><p>Position-aware Transformer · matched frozen/full/LoRA branches</p><p><b>Independent generalization: NOT ACHIEVED. External deployment prohibited.</b></p><p>All results come from public synthetic recipes. Five initialization seeds measure training variability; they do not establish independent task-family sampling.</p><table><tr><th>Branch</th><th>Seeds</th><th>Mean accuracy</th><th>Range</th><th>Complete-pair accuracy</th></tr>'+body+'</table><p>Per-family, per-category, class coverage, counterfactuals and resource measurements are included in the JSON report. Failed and interrupted experiments remain recorded.</p><img width="100%" src="SAAL-RSI-v0.9.3-Comparison.svg"></html>'
(out/'SAAL-RSI-v0.9.3-Research-Results.html').write_text(page)
print(json.dumps({'report':str(out/'SAAL-RSI-v0.9.3-Research-Results.html'),'rows':rows}))
