"""Aggregate resource samples; RSS sums are not unique physical memory."""
import datetime
import json
import os
import re
import subprocess
import time
import urllib.request

REVISION='phase-page-delta-v2'


def snapshot(phase='unspecified',previous=None,expected_interval_seconds=None):
    monotonic=time.monotonic()
    vm=subprocess.check_output(['vm_stat'],text=True)
    counts={}
    for key in ['Swapins','Swapouts','Pageins','Pageouts','Pages occupied by compressor']:
        match=re.search(r'^'+re.escape(key)+r':\s+(\d+)',vm,re.M)
        if match:counts[key]=int(match[1])
    process_rows=subprocess.check_output(['ps','-axo','rss,pcpu,comm'],text=True).splitlines()[1:]
    own=[];other=[]
    for row in process_rows:
        fields=row.split(None,2)
        if len(fields)==3:
            (own if 'ollama' in fields[2].lower() else other).append((int(fields[0]),float(fields[1])))
    page_size=os.sysconf('SC_PAGE_SIZE')
    value={'sampling_revision':REVISION,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monotonic_seconds':monotonic,'phase':phase,'page_size_bytes':page_size,'ollama_aggregate_rss_kib':sum(r[0] for r in own),'vm_counters':counts,'swap':subprocess.check_output(['sysctl','vm.swapusage'],text=True).strip(),'memory_pressure':subprocess.check_output(['memory_pressure','-Q'],text=True).splitlines()[-1],'kernel_pressure_level':subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True).strip(),'loadavg_1_5_15':list(os.getloadavg()),'other_workloads':{'process_count':len(other),'aggregate_rss_kib_not_unique_physical_memory':sum(r[0] for r in other),'aggregate_cpu_percent':sum(r[1] for r in other)}}
    ps=json.load(urllib.request.urlopen('http://127.0.0.1:11434/api/ps',timeout=2))
    value['models']=[{k:m.get(k) for k in ['name','digest','size','size_vram','context_length','expires_at']} for m in ps['models']]
    if previous:
        interval=monotonic-previous['monotonic_seconds']
        value['sample_interval_seconds']=interval
        value['late_sample']=expected_interval_seconds is not None and interval>expected_interval_seconds*1.5
        value['counter_deltas']={}
        for key in ['Swapins','Swapouts','Pageins','Pageouts']:
            pages=counts[key]-previous['vm_counters'][key]
            value['counter_deltas'][key]={'pages':pages,'mib':pages*page_size/1048576,'mib_per_second':pages*page_size/1048576/interval,'counter_reset':pages<0}
    return value

if __name__=='__main__':
    value=snapshot()
    with open('eval/resource-samples.jsonl','a') as out:out.write(json.dumps(value)+'\n')
    print(json.dumps(value))
