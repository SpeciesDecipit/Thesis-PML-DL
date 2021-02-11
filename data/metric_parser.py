import os
import csv
from shutil import copy2

SEEMS_TO_BE = '{} {} seems to be a {}'

KIND = 'Kind'
NAME = 'Name'
FILE = 'File'

LOC = 'CountLineCode'
DIT = 'MaxInheritanceTree'
RFC = 'CountDeclMethodAll'
WMC = 'SumCyclomatic'
LCOM = 'PercentLackOfCohesion'
NOC = 'CountClassDerived'
CBO = 'CountClassCoupled'

LARGE_CLASS_PATH = './large_class/'
LAZY_CLASS_PATH = './lazy_class/'
DATA_CLASS_PATH = './data_class/'
PARALLEL_INHERITANCE_HIERARCHIES_PATH = './parallel_inheritance_hierarchies/'
GOD_CLASS_PATH = './god_class/'
FEATURE_ENVY_PATH = './feature_envy/'


def copy_file(src_file_path, dst_dir_path):
    os.makedirs(dst_dir_path, exist_ok=True)
    copy2(src_file_path, dst_dir_path)


def check_condition(row, key, condition):
    return row[key] and condition(int(row[key]))


def check_large_class(row):
    if (check_condition(row, LOC, lambda x: x > 300)
            or check_condition(row, DIT, lambda x: x > 5)
            or check_condition(row, RFC, lambda x: x > 20)):
        print(SEEMS_TO_BE.format(row[KIND], row[NAME], 'Large Class'))
        print(f'{LOC} = {row[LOC]}, {DIT} = {row[DIT]}, {RFC} = {row[RFC]}')
        copy_file(row[FILE], LARGE_CLASS_PATH)


def check_lazy_class(row):
    if (check_condition(row, RFC, lambda x: x == 0)
            or check_condition(row, LOC, lambda x: x < 100)
            or check_condition(row, WMC, lambda x: x <= 2)):
        print(SEEMS_TO_BE.format(row[KIND], row[NAME], 'Lazy Class'))
        print(f'{RFC} = {row[RFC]}, {LOC} = {row[LOC]}, {WMC} = {row[WMC]}')
        copy_file(row[FILE], LAZY_CLASS_PATH)


def check_data_class(row):
    if (check_condition(row, LCOM, lambda x: x > 80)
            or check_condition(row, WMC, lambda x: x > 50)):
        print(SEEMS_TO_BE.format(row[KIND], row[NAME], 'Data Class'))
        print(f'{LCOM} = {row[LCOM]}, {WMC} = {row[WMC]}')
        copy_file(row[FILE], DATA_CLASS_PATH)


def check_parallel_inheritance_hierarchies(row):
    if (check_condition(row, DIT, lambda x: x > 3)
            or check_condition(row, NOC, lambda x: x > 4)):
        print(SEEMS_TO_BE.format(row[KIND], row[NAME],
                                 'Parallel Inheritance Hierarchies'))
        print(f'{DIT} = {row[DIT]}, {NOC} = {row[NOC]}')
        copy_file(row[FILE], PARALLEL_INHERITANCE_HIERARCHIES_PATH)


def check_god_class(row):
    if check_condition(row, WMC, lambda x: x > 47):
        print(SEEMS_TO_BE.format(row[KIND], row[NAME], 'God Class'))
        print(f'{WMC} = {row[WMC]}')
        copy_file(row[FILE], GOD_CLASS_PATH)


def check_feature_envy(row):
    if (check_condition(row, CBO, lambda x: x > 5)
            or check_condition(row, LCOM, lambda x: x > 50)):
        print(SEEMS_TO_BE.format(row[KIND], row[NAME], 'Feature Envy'))
        print(f'{CBO} = {row[CBO]}, {LCOM} = {row[LCOM]}')
        copy_file(row[FILE], FEATURE_ENVY_PATH)


file_path = input()

with open(file_path) as csv_file:
    csv_reader = csv.DictReader(csv_file)
    print(f'Column names are {csv_reader.fieldnames}')
    for row in csv_reader:
        if 'Class' not in row[KIND]:
            continue
        # check_large_class(row)
        # check_lazy_class(row)
        check_data_class(row)
        check_parallel_inheritance_hierarchies(row)
        check_god_class(row)
        check_feature_envy(row)
