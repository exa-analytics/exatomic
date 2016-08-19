#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2016, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
Build/Publish
#################
"""
import os
import shutil
import argparse
import subprocess


builds = ["win-32", "win-64", "osx-64", "linux-32", "linux-64"]


def get_args():
    """
    Description of the deployment script.
    """
    parser = argparse.ArgumentParser(description="Deployment script for Exa Analytics")
    parser.add_argument("pkg", nargs="?")
    parser.add_argument("version", nargs="?")
    parser.add_argument("which_pypi", nargs="?", default="pypi")
    return parser.parse_args()


def get_package_name():
    """
    Automatically determine the package name.
    """
    for item in os.listdir("."):
        if item.startswith("ex") and "." not in item:
            return item
    raise Exception("Package name unknown")


def get_package_version(pkg):
    """
    Determine the package version from the primary __init__.py file.
    """
    search = "__{}_version__".format(pkg)
    versions = []
    with open(os.sep.join((pkg, "__init__.py"))) as f:
        for line in f:
            if search in line:
                v = line.split('=')[-1]
                v = v.replace("(", "").replace(")", "").replace(",", " ")
                versions.append(tuple((int(i) for i in v.split())))
                break
    with open("meta.yaml") as f:
        for line in f:
            if any(x in line for x in ["version:", "git_rev:"]):
                v = line.split()[-1].replace('"', '').replace('v', '').split('.')
                versions.append(tuple(int(i) for i in v))
    if not all(version == versions[0] for version in versions):
        raise Exception("Version mismatch between init {} and meta.yaml {}!".format(versions[0], versions[1]))
    else:
        return versions[0]


def get_python_versions():
    """
    Get versions of python to build for.
    """
    versions = []
    read = False
    with open(".travis.yml") as f:
        for line in f:
            if "python:" in line:
                read = True
            elif "-" not in line and read:
                read = False
            elif read:
                versions.append(line.split()[-1].replace('"', ''))
    return versions


def check(pkg, version, pys):
    """
    Human check for publication.
    """
    v = ".".join([str(i) for i in version])
    py = ", ".join([py.replace("'", "") for py in pys])
    print("Before deploying, make sure '{}' version {} has been tagged for release.".format(pkg, v))
    inp = "Deploy '{}' {} for python {} on all platforms (y/N): ".format(pkg, v, py)
    chk = input(inp)
    if chk.lower() == "y":
        return True
    return False


def conda_build(pys):
    """
    Deploy to anaconda.
    """
    build_fmt = "conda build --python={py} ."
    conv_fmt = "conda convert {} -p all"
    conup_fmt = "anaconda upload {}/{}"
    for py in pys:
        string = build_fmt.format(py=py)
        build = subprocess.run([string], shell=True, check=True, stdout=subprocess.PIPE)
        if build.returncode != 0:
            raise Exception("Build failed with command {}".format(string))
        for line in build.stdout.decode("utf-8").split('\n'):
            if "anaconda upload" in line:
                path = line.split()[-1]
        string = conv_fmt.format(path)
        conv = subprocess.run([string], shell=True, check=True, stdout=subprocess.PIPE)
        if conv.returncode != 0:
            raise Exception("Convert failed with command {}".format(string))
        name = path.split(os.sep)[-1]
        for pltfrm in builds:
            string = conup_fmt.format(pltfrm, name)
            upload = subprocess.run([string], shell=True, check=True,
                                    stdout=subprocess.PIPE, stdin=subprocess.PIPE)
            if upload.returncode != 0:
                raise Exception("Upload failed with command {}".format(string))


def pypi_build(pypi):
    """
    Deploy to pypi/pypitest.
    """
    string = "python setup.py sdist upload {}".format(pypi)
    ret = subprocess.run([string], shell=True, check=True,
                         stdout=subprocess.PIPE, stdin=subprocess.PIPE)


def cleanup():
    """
    Cleanup the build directories.
    """
    for build in builds:
        if os.path.isdir(build):
            shutil.rmtree(build)


if __name__ == "__main__":
    args = get_args()
    if args.pkg is None:
        args.pkg = get_package_name()
    if args.version is None:
        args.version = get_package_version(args.pkg)
    pys = get_python_versions()
    chk = check(args.pkg, args.version, pys)
    if chk:
        conda_build(pys)
        pypi_build(args.which_pypi)
    cleanup()
