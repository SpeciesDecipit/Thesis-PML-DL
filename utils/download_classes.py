import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List

from repo_urls import systems_git

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)


@dataclass
class RepoDetail:
    name: str
    url: str
    snapshot: str
    directory: List[str]


for repo in systems_git:
    repo_detail = RepoDetail(**repo)

    TEMP = os.path.join(ROOT_DIR, 'data_generation', 'TEMP')
    repo_dir = os.path.join(TEMP, repo_detail.name)

    if not os.path.exists(TEMP):
        os.makedirs(TEMP)

    # Clone if not exists
    if not os.path.exists(repo_dir):
        cloneCommand = 'git clone ' + repo_detail.url + ' ' + repo_dir
        subprocess.call(cloneCommand, shell=True)

    path_to_sources = repo_dir + repo_detail.directory[0]
    path_to_classes = ROOT_DIR + '/' + 'antipatterns/' + repo_detail.name + '/god_class.txt'
    output = ROOT_DIR + '/god_classes'

    for path_to_god_class in open(path_to_classes, 'r').readlines():
        abs_path_to_god_class = (
            path_to_sources + path_to_god_class.replace('.', '/').replace('\n', '') + '.java'
        )
        subprocess.call(['cp', abs_path_to_god_class, output])
        open(abs_path_to_god_class, 'r').close()

