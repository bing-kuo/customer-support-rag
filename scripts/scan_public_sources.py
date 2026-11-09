"""Stage allowlisted public files without opening local secrets/private evidence."""
import hashlib,json,os,shutil,stat,subprocess,tempfile
from pathlib import Path

SCAN_POLICY='allowlisted-public-sources-v2-no-local-secret-read'
IMAGE='ghcr.io/gitleaks/gitleaks@sha256:e1b35e12a8c6fa8901f060459cfb6b2fc4c484d3afbe3b029733a3bbfab07055'
ROOT_FILES={'.gitignore','.gitleaks.toml','AGENTS.md','README.md','LICENSE','NOTICE','requirements.lock'}
ROOT_DIRS={'.github','app','tests','scripts','web','fixtures','docs','licenses','notices','knowledge'}


def stage_sources(root,target):
    root=Path(root);target=Path(target);files={}
    for base,dirs,names in os.walk(root,followlinks=False):
        relative=Path(base).relative_to(root)
        if relative==Path('.'):
            dirs[:]=[d for d in dirs if d in ROOT_DIRS and not (Path(base)/d).is_symlink()]
        elif relative==Path('knowledge'):
            dirs[:]=[d for d in dirs if d=='documents' and not (Path(base)/d).is_symlink()]
        else:dirs[:]=[d for d in dirs if d not in ('__pycache__','.git','.venv','runtime') and not (Path(base)/d).is_symlink()]
        for name in names:
            if relative==Path('knowledge'):continue
            if relative==Path('.') and name not in ROOT_FILES:continue
            if name=='.env' or name.startswith('.env.') or name=='project.env' or '.sqlite3' in name or name.endswith('.pyc'):continue
            source=Path(base)/name
            if source.is_symlink():continue
            descriptor=os.open(source,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
            try:
                meta=os.fstat(descriptor)
                if not stat.S_ISREG(meta.st_mode) or meta.st_size>2**22:raise ValueError('Public scan boundary exceeded')
                with os.fdopen(descriptor,'rb',closefd=False) as stream:data=stream.read()
            finally:os.close(descriptor)
            key=str(source.relative_to(root));dest=target/key;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);files[key]=hashlib.sha256(data).hexdigest()
    return {'policy':SCAN_POLICY,'files':files,'excluded_categories':['project secrets','private runtime/ledger','all eval raw/private evidence','git/environment caches','symlinks']}


def scan(root,out):
    with tempfile.TemporaryDirectory(prefix='public-source-scan-') as temporary:
        manifest=stage_sources(root,temporary)
        (out/'scan-manifest.json').write_text(json.dumps(manifest,sort_keys=True,separators=(',',':')))
        result=subprocess.run(['docker','run','--pull=never','--rm','-v',temporary+':/repo:ro','-v',str(out)+':/reports',IMAGE,'dir','/repo','--config','/repo/.gitleaks.toml','--report-format','json','--report-path','/reports/gitleaks.json','--redact','--no-banner'],check=False)
        return result.returncode
