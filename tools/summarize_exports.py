"""仅解析历史 SwanLab 导出，不加载模型或执行训练。"""
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
results = []
for path in sorted((ROOT / 'experiments/logs').glob('*.csv')):
    grouped = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            grouped.setdefault(row['key'], []).append(row)
    metrics = {}
    for key in ['loss', 'logits_loss', 'aux_loss', 'learning_rate', '__swanlab__.gpu.0.mem.value']:
        rows = grouped[key]
        values = [float(r['value']) for r in rows]
        metrics[key] = dict(count=len(rows), first=values[0], last=values[-1], minimum=min(values), maximum=max(values), first_time=rows[0]['time_local'], last_time=rows[-1]['time_local'])
    times = [datetime.fromisoformat(r['time_utc']) for r in grouped['__swanlab__.gpu.0.mem.value']]
    interval = statistics.median((b-a).total_seconds() for a,b in zip(times,times[1:]))
    console = path.with_name(path.stem + '_console.txt')
    text = console.read_text(encoding='utf-8-sig')
    progress = re.findall(r'Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\), loss: ([\d.]+)', text)
    if not progress:
        raise ValueError(f'控制台缺少训练进度: {console.name}')
    epoch, epochs, step, steps, loss = progress[-1]
    assert epoch == epochs and step == steps, f'最后进度未到训练终点: {path.name}'
    assert abs(float(loss) - metrics['loss']['last']) < 0.0001
    results.append(dict(run=path.stem, metrics=metrics, memory_interval_median_seconds=interval, console_final_epoch=int(epoch), console_steps_per_epoch=int(steps), dataloader_total_steps=int(epochs)*int(steps), sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [path,console]}))
output = ROOT / 'experiments/summary.json'
output.write_text(json.dumps({'source':'原始 CSV 与控制台日志；CSV step 为各指标记录索引，训练步数取控制台', 'runs':results}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(f'已核对 {len(results)} 次训练，生成 experiments/summary.json')
