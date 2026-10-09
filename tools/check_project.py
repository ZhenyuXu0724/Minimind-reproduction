"""Check source syntax, archived results and published documentation without ML dependencies."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for path in list((ROOT / 'code').rglob('*.py')) + list((ROOT / 'tools').glob('*.py')):
    ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
for path in (ROOT / 'code/model').glob('*.json'):
    json.loads(path.read_text(encoding='utf-8-sig'))

markdown = [ROOT / 'README.md', *list((ROOT / 'docs').glob('*.md')), *list((ROOT / 'configs').glob('*.md'))]
for path in markdown:
    text = re.sub(r'```.*?```', '', path.read_text(encoding='utf-8'), flags=re.S)
    for target in re.findall(r'\]\(([^\n]+?)\)', text):
        target = target.strip('<>')
        if target.startswith(('https://', 'http://', '#')):
            continue
        assert (path.parent / target).exists(), f'Broken link: {path.relative_to(ROOT)} -> {target}'

config = json.loads((ROOT / 'configs/historical_runs.json').read_text(encoding='utf-8'))
summary = json.loads((ROOT / 'experiments/summary.json').read_text(encoding='utf-8'))
assert len(config['runs']) == len(summary['runs']) == 3
for historical, actual in zip(config['runs'], summary['runs']):
    for field, stat in [('loss_first', 'first'), ('loss_last', 'last'), ('loss_min', 'minimum')]:
        assert abs(historical[field] - actual['metrics']['loss'][stat]) < 0.0001
    assert historical['dataloader_steps'] == actual['dataloader_total_steps']
    assert historical['sampled_peak_gpu_memory_mb'] == actual['metrics']['__swanlab__.gpu.0.mem.value']['maximum']

print('Source syntax, tokenizer JSON, local links and historical metrics validated.')
