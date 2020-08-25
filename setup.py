#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, platform
from distutils import log
from subprocess import check_call
from setuptools import setup, find_packages, Command
from setuptools.command.sdist import sdist
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info


name = "exatomic"
description = "A unified platform for theoretical and computational chemists."
staticdir = "static"
jsdir = "js"
jsroot = os.path.abspath(jsdir)
readme = "README.md"
requirements = "requirements.txt"
log.set_verbosity(log.DEBUG)
root = os.path.dirname(os.path.abspath(__file__))
node_modules = os.path.join(root, jsdir, "node_modules")
prckws = {'shell': True} if platform.system().lower() == "windows" else {}
npm_path = os.pathsep.join([os.path.join(node_modules, ".bin"),
                            os.environ.get("PATH", os.defpath)])
try:
    import pypandoc
    long_description = pypandoc.convert(readme, "rst")
except ImportError:
    with open(readme) as f:
        long_description = f.read()
with open(requirements) as f:
    dependencies = f.read().splitlines()
with open(os.path.abspath(os.path.join(os.path.dirname(__file__), name, "static", "version.txt"))) as f:
    __version__ = f.read().strip()


def update_package_data(distribution):
    """Modify the ``package_data`` to catch changes during setup."""
    build_py = distribution.get_command_obj("build_py")
    build_py.finalize_options()    # Updates package_data


def js_prerelease(command, strict=False):
    """Build minified JS/CSS prior to performing the command."""
    class DecoratedCommand(command):
        """
        Used by ``js_prerelease`` to modify JS/CSS prior to running the command.
        """
        def run(self):
            jsdeps = self.distribution.get_command_obj("jsdeps")
            if not os.path.exists(".git") and all(os.path.exists(t) for t in jsdeps.targets):
                command.run(self)
                return
            try:
                self.distribution.run_command("jsdeps")
            except Exception as e:
                missing = [t for t in jsdeps.targets if not os.path.exists(t)]
                if strict or missing:
                    log.warn("Rebuilding JS/CSS failed")
                    if missing:
                        log.error("Missing files: {}".format(missing))
                    raise e
                else:
                    log.warn("Rebuilding JS/CSS failed but continuing...")
                    log.warn(str(e))
            command.run(self)
            update_package_data(self.distribution)
    return DecoratedCommand


class NPM(Command):
    description = "install package.json dependencies using npm."
    user_options = []
    targets = [os.path.join(root, name, staticdir, "js", "extension.js"),
               os.path.join(root, name, staticdir, "js", "index.js")]

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
        #package_json = os.path.join(jsroot, "package.json")
        #node_modules_exists = os.path.exists(node_modules)
        return self.has_npm()

    def run(self):
        if not self.has_npm():
            log.error("`npm` unavailable.  If you're running this command using sudo, make sure `npm` is available to sudo")
        else:
            env = os.environ.copy()
            env['PATH'] = npm_path
            log.info("Installing build dependencies with npm. This may take a while...")
            check_call(["npm", "install"], cwd=jsroot, stdout=sys.stdout, stderr=sys.stderr, **prckws)
            os.utime(node_modules, None)

        for t in self.targets:
            if not os.path.exists(t):
                msg = "Missing file: %s" % t
                if not self.has_npm():
                    msg += "\nnpm is required to build a development version of " + name
                raise ValueError(msg)

        # update package data in case this created new files
        update_package_data(self.distribution)


setup_args = {
    'name': name,
    'version': __version__,
    'description': description,
    'long_description': long_description,
    'package_data': {name: [staticdir + "/*"]},
    'include_package_data': True,
    'install_requires': dependencies,
    'packages': find_packages(),
    'data_files': [
        ("share/jupyter/nbextensions/" + name, [
            os.path.join(name, staticdir, "js", "extension.js"),
            os.path.join(name, staticdir, "js", "index.js"),
            os.path.join(name, staticdir, "js", "index.js.map")
        ]),
    ],
    'cmdclass': {
        'build_py': js_prerelease(build_py),
        'egg_info': js_prerelease(egg_info),
        'sdist': js_prerelease(sdist, strict=True),
        'jsdeps': NPM,
    },
    'zip_safe': False,
    'license': "Apache License Version 2.0",
    'author': "Thomas J. Duignan, Alex Marchenko and contributors",
    'author_email': "exa.data.analytics@gmail.com",
    'maintainer_email': "exa.data.analytics@gmail.com",
    'url': f"https://github.com/exa-analytics/{name}",
    'download_url': f"https://pypi.io/packages/source/e/{name}/{name}-{__version__}.tar.gz",
    'keywords': ["quantum chemistry", "jupyter notebook", "visualization"],
    'classifiers': [
        "Development Status :: 4 - Beta",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Natural Language :: English"
    ]
}

setup(**setup_args)
