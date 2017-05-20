#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
from __future__ import print_function
from setuptools import setup, find_packages, Command
from setuptools.command.sdist import sdist
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info
from subprocess import check_call
from distutils import log
import os
import sys
import platform


name = "exatomic"
description = "A unified platform for theoretical and computational chemists and physicists."
jsbuilddir = os.path.join('exatomic', '_nbextension')
here = os.path.dirname(os.path.abspath(__file__))
node_root = os.path.join(here, "js")
is_repo = os.path.exists(os.path.join(here, ".git"))
prckws = {'shell': True} if platform.system().lower() == 'windows' else {}
log.set_verbosity(log.DEBUG)
log.info("setup.py entered")
log.info("$PATH=%s" % os.environ['PATH'])
npm_path = os.pathsep.join([os.path.join(node_root, "node_modules", ".bin"),
                            os.environ.get('PATH', os.defpath)])


try:
    import pypandoc
    long_description = pypandoc.convert("README.md", "rst")
except ImportError:
    with open("README.md") as f:
        long_description = f.read()
with open("requirements.txt") as f:
    dependencies = f.read().splitlines()
with open(os.path.join(here, name, "_version.py")) as f:
    v = f.readlines()[-2]
    v = v.split('=')[1].strip()[1:-1]
    version = ".".join(v.replace(" ", "").split(","))


def js_prerelease(command, strict=False):
    """Decorator for building minified js/css prior to another command."""
    class DecoratedCommand(command):
        def run(self):
            jsdeps = self.distribution.get_command_obj('jsdeps')
            if not is_repo and all(os.path.exists(t) for t in jsdeps.targets):
                # sdist, nothing to do
                command.run(self)
                return
            try:
                self.distribution.run_command('jsdeps')
            except Exception as e:
                missing = [t for t in jsdeps.targets if not os.path.exists(t)]
                if strict or missing:
                    log.warn("rebuilding js and css failed")
                    if missing:
                        log.error("missing files: %s" % missing)
                    raise e
                else:
                    log.warn("rebuilding js and css failed (not a problem)")
                    log.warn(str(e))
            command.run(self)
            update_package_data(self.distribution)
    return DecoratedCommand


def update_package_data(distribution):
    """Update package_data to catch changes during setup."""
    build_py = distribution.get_command_obj('build_py')
    # distribution.package_data = find_package_data()
    # re-init build_py options which load package_data
    build_py.finalize_options()


class NPM(Command):
    description = "Install package.json dependencies using npm."
    user_options = []
    node_modules = os.path.join(node_root, "node_modules")
    targets = [os.path.join(here, jsbuilddir, "extension.js"),
               os.path.join(here, jsbuilddir, "index.js")]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def has_npm(self):
        try:
            check_call(["npm", "--version"], **prckws)
            return True
        except Exception:
            return False

    def should_run_npm_install(self):
        #package_json = os.path.join(node_root, "package.json")
        #node_modules_exists = os.path.exists(self.node_modules)
        return self.has_npm()

    def run(self):
        has_npm_ = self.has_npm()
        if not has_npm_:
            log.error("`npm` unavailable.  If you're running this command using sudo, make sure `npm` is available to sudo")

        env = os.environ.copy()
        env['PATH'] = npm_path

        if self.should_run_npm_install():
            log.info("Installing build dependencies with npm. This may take a while...")
            check_call(["npm", "install"], cwd=node_root, stdout=sys.stdout, stderr=sys.stderr, **prckws)
            os.utime(self.node_modules, None)

        for t in self.targets:
            if not os.path.exists(t):
                msg = "Missing file: %s" % t
                if not has_npm_:
                    msg += "\nnpm is required to build a development version of widgetsnbextension"
                raise ValueError(msg)

        # update package data in case this created new files
        update_package_data(self.distribution)


setup_args = {
    'name': name,
    'version': version,
    'description': description,
    'long_description': long_description,
    'include_package_data': True,
    'data_files': [
        ("share/jupyter/nbextensions/jupyter-" + name, [
            os.path.join(jsbuilddir, "extension.js"),
            os.path.join(jsbuilddir, "index.js"),
            os.path.join(jsbuilddir, "index.js.map")
        ]),
    ],
    'install_requires': dependencies,
    'packages': find_packages(),
    'zip_safe': False,
    'cmdclass': {
        'build_py': js_prerelease(build_py),
        'egg_info': js_prerelease(egg_info),
        'sdist': js_prerelease(sdist, strict=True),
        'jsdeps': NPM,
    },
    'license': "Apache License Version 2.0",
    'author': "Thomas J. Duignan and Alex Marchenko",
    'author_email': "exa.data.analytics@gmail.com",
    'maintainer_email': "exa.data.analytics@gmail.com",
    'url': "https://exa-analytics.github.io/" + name,
    'download_url': "https://github.com/exa-analytics/" + name + "/tarball/v{}.tar.gz".format(version),
    'keywords': ['visualization'],
    'classifiers': [
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Legal Industry",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English"
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Multimedia :: Graphics"
    ]
}

setup(**setup_args)
