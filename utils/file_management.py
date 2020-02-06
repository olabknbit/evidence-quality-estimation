import os
import os.path
from os import listdir
from os.path import isfile, isdir, join
from typing import List


def save_data_with_ultimate_dir_creation(path: str, lines: List[str]) -> None:
    elems = path.split('/')
    for i in range(len(elems))[2:]:
        directory = '/' + os.path.join(*elems[:i]) if elems[1] == 'Users' else os.path.join(*elems[:i])

        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError:
                print("Creation of directory %s failed" % directory)
            else:
                print("Successfully created directory %s " % directory)

    # save data to files
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def save_data_with_dir_creation(directory: str, filename: str, lines: List[str]) -> None:
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError:
            print("Creation of directory %s failed" % directory)
        else:
            print("Successfully created directory %s " % directory)

    # save data to files
    with open(os.path.join(directory, filename), 'w', encoding='utf-8') as f:
        f.writelines(lines)


def get_all_dirnames(directory: str) -> List[str]:
    onlydirs = [d for d in listdir(directory) if isdir(join(directory, d))]
    return onlydirs


def get_all_dirpaths(directory: str) -> List[str]:
    onlydirs = [join(directory, d) for d in get_all_dirnames(directory)]
    return onlydirs


def get_all_filenames(directory: str) -> List[str]:
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    return onlyfiles


def get_all_filepaths(directory: str) -> List[str]:
    onlyfiles = [join(directory, f) for f in get_all_filenames(directory)]
    return onlyfiles


def get_all_filepaths_rec(directory: str) -> List[str]:
    paths = get_all_filepaths(directory)
    for dir_path in get_all_dirpaths(directory):
        paths = paths + get_all_filepaths_rec(dir_path)
    return paths


def count_num_of_files_in_respective_dirs(directory):
    for dirpath in get_all_dirpaths(directory):
        tt = dirpath[len(directory):]
        for dirpath_tt in get_all_dirpaths(dirpath):
            feature_name = dirpath[len(dirpath_tt):]

            for dirpath2 in get_all_dirpaths(dirpath_tt):
                classifier_name = dirpath2[len(dirpath) + 1:]

                a = len(get_all_filenames(dirpath2 + '/A/'))
                b = len(get_all_filenames(dirpath2 + '/B/'))
                c = len(get_all_filenames(dirpath2 + '/C/'))
                print('A:\t', a, '\tB:\t', b, '\tC:\t', c, '\tall:\t', a + b + c, feature_name, '\t', classifier_name,
                      '\t', tt)


def get_all_filepaths_in_dirs(directories: List[str]) -> List[str]:
    all_paths = []
    for directory in directories:
        paths = get_all_filepaths_rec(directory)
        all_paths.extend(paths)
    return all_paths
