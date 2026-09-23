"""Create a private GitHub repository only after local checks and explicit authorization.

GitHub CLI performs authentication. This program never accepts or stores tokens.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
REPOSITORY='DrCanD/spikescr-local-event-geometry'


def run(args, capture=False, check=True):
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    return subprocess.run(args,cwd=ROOT,env=env,text=True,capture_output=capture,check=check)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--create',action='store_true',help='Authorize private repository creation and push')
    a=p.parse_args()
    from ssc_geometry.integrity import verify_integrity
    verify_integrity(ROOT)
    if not a.create:
        print('No remote changes made. Use --create to authorize creation of the private repository '+REPOSITORY)
        return
    if not shutil.which('git') or not shutil.which('gh'):
        raise SystemExit('Git and the GitHub CLI must be installed. Authenticate with gh auth login, then rerun this command. Do not put a token in this directory.')
    run(['gh','auth','status'])
    login=run(['gh','api','user','--jq','.login'],capture=True).stdout.strip()
    if login.lower()!='drcand':raise SystemExit('Authenticated GitHub account is not DrCanD; no repository was created.')
    exists=run(['gh','repo','view',REPOSITORY,'--json','nameWithOwner'],capture=True,check=False)
    if exists.returncode==0:raise SystemExit('The repository already exists. Nothing was overwritten. Inspect it before deciding how to push.')
    run([sys.executable,'-m','unittest','discover','-s','tests','-v'])
    if not (ROOT/'.git').exists():
        run(['git','init','-b','main'])
        # Stage only the audited manifest, never an unrelated top-level file.
        manifest=json.loads((ROOT/'checksums/manifest.json').read_text(encoding='utf-8'))
        reviewed=sorted(manifest['files'])+['checksums/manifest.json','checksums/SHA256SUMS.txt']
        run(['git','add','--',*reviewed])
        run(['git','-c','user.name=Ismail Can Dikmen','-c','user.email=can.dikmen@istinye.edu.tr','commit','-m','Prepare audited fixed-checkpoint reproduction package'])
    if run(['git','status','--porcelain'],capture=True).stdout.strip():
        raise SystemExit('Working tree is not clean. Commit reviewed changes before publication.')
    if run(['git','branch','--show-current'],capture=True).stdout.strip()!='main':
        raise SystemExit('The checked publication branch must be main.')
    if run(['git','remote'],capture=True).stdout.strip():
        raise SystemExit('A remote is already configured. Nothing was changed; inspect the remote explicitly.')
    local_sha=run(['git','rev-parse','HEAD'],capture=True).stdout.strip()
    run(['gh','repo','create',REPOSITORY,'--private','--source',str(ROOT),'--remote','origin','--push',
         '--description','Fixed-checkpoint SpikeSCR local event geometry and reproducibility checks'])
    remote=json.loads(run(['gh','repo','view',REPOSITORY,'--json','nameWithOwner,isPrivate,url'],capture=True).stdout)
    remote_sha=run(['gh','api',f'repos/{REPOSITORY}/commits/main','--jq','.sha'],capture=True).stdout.strip()
    if remote.get('nameWithOwner')!=REPOSITORY or remote.get('isPrivate') is not True or remote_sha!=local_sha:
        raise SystemExit('Post-publication verification failed. Inspect the remote; do not claim a verified release.')
    print(json.dumps({'status':'CREATED_AND_COMMIT_VERIFIED','repository':remote,'commit':local_sha,
        'public_release_created':False,'archival_doi_created':False},indent=2))
if __name__=='__main__':main()
