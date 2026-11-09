"""Resource evidence unit regression for actual page units and sampling gaps."""
import io
import json
from types import SimpleNamespace
from pathlib import Path


def test_swap_page_conversion_and_gap_are_measured_not_assumed(monkeypatch):
    from scripts import resource_snapshot as module
    value={'pages':877008,'clock':10.0}
    def command(cmd,**kwargs):
        if cmd[0]=='vm_stat':return '\n'.join(f'{name}: {value["pages"]}.' for name in ['Swapins','Swapouts','Pageins','Pageouts','Pages occupied by compressor'])
        if cmd[0]=='ps':return 'RSS %CPU COMM\n100 2.0 ollama\n200 3.0 another-app\n'
        if cmd[0]=='memory_pressure':return 'System-wide memory free percentage: 20%'
        if cmd==['sysctl','-n','kern.memorystatus_vm_pressure_level']:return '2\n'
        return 'vm.swapusage: total = 8192.00M used = 7003.31M'
    monkeypatch.setattr(module.subprocess,'check_output',command)
    monkeypatch.setattr(module,'os',SimpleNamespace(sysconf=lambda _:16384,getloadavg=lambda :(1,2,3)))
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda :value['clock']))
    monkeypatch.setattr(module.urllib.request,'urlopen',lambda *args,**kwargs:io.BytesIO(b'{"models":[]}'))
    first=module.snapshot('before')
    value.update(pages=902656,clock=20.0)
    second=module.snapshot('generation',first,2)
    assert second['counter_deltas']['Swapouts']=={'pages':25648,'mib':400.75,'mib_per_second':40.075,'counter_reset':False}
    assert second['sample_interval_seconds']==10 and second['late_sample'] is True
    assert first['phase']=='before' and second['kernel_pressure_level']=='2'
    assert second['page_size_bytes']==16384 and second['ollama_aggregate_rss_kib']==100
    assert second['other_workloads']['aggregate_rss_kib_not_unique_physical_memory']==200


def test_public_resource_subset_is_deterministic_read_only_and_has20warm(monkeypatch):
    from scripts import resource_trial as module
    from app.knowledge import digest
    folder=Path(__file__).resolve().parents[1]/'eval'/'frozen-v3'
    monkeypatch.setattr(module,'FROZEN',folder)
    before=(folder/'dev.json').read_bytes()
    subset=module.frozen_subset()
    assert subset==module.frozen_subset() and len(subset['cases'])==21
    assert len({q['id'] for q in subset['cases']})==21
    assert all(q['expected']=='answer' and q['gold'] and not q.get('prior_turns') for q in subset['cases'])
    assert subset['dev_raw_sha256']==digest(before)
    assert (folder/'dev.json').read_bytes()==before
