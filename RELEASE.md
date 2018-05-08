# Update Version Numbers
Update `exatomic/_version.py`, js/package.json` and `meta.yaml` version numbers.


# Commit and Push Changes
Push changes to your repository (fork).
```bash
git add -A
git commit -m "message"
git push
```
Make a pull request (PR). Once accepted the organization will be tag a new release.
```bash
git tag -a X.X.X -m "message"
git push --tags
```


# Remove existing builds (if present)
```bash
rm -r dist/*
```


# Release on PyPI
Publish to `TestPyPI`_ (~/.pypirc required).
This requires `wheel` and `twine` to be installed.
```bash
python setup.py sdist    #python setup.py bdist_wheel
twine upload dist/*
```
The variable `which` refers to the alias for the testing or production
repository listed in ~/.pypirc.


# Release on NPM
Publish the JavaScript extensions on NPM (if required).
In the JavaScript directory execute the following commands.
```bash
npm whoami   # Confirm correct user if incorrect change with npm adduser, npm login
npm publish
```


# Release on Anaconda
Publish to the `exa-analytics`_ channel on  `Anaconda Cloud`.
This requires that the package hase been released on `PyPI`.
This requires the anaconda package `conda-build` and `anaconda-client` (run `anaconda login`).
It is convenient to use a `.condarc` file to add the `conda-forge` and 
`exaanalytics` channels since some auxiliary packages come from there.
Note that to test a build, set uploading to false (anaconda_upload: false in .condarc) and
in the meta.yaml source change the url to `git_url: ./`.
```bash
conda build .    # conda build . --output to see location
# For other python version, conda build --python x.x
conda convert -f --platform all /path/to/conda-bld/pltfrm/exa-...tar.bz2 -o /path/to/outputdir/
anaconda upload /path/to/build/build.tar.bz2    # For each build
```
