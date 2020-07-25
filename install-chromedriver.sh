#!/usr/bin/env bash

echo """
Installer script to set up chromedriver for selenium
on WSL. This allows for running selenium automated
browser tests.

Needs approximately the following in the bash environment:

    export DISPLAY=:0
    export BROWSER=/mnt/c/Program\ Files\ \(x86\)/Google/Chrome/Application/chrome.exe

Depending on chrome version, please update CHROME_SUFFIX.
If on latest, CHROME_SUFFIX can be "".
"""

CHROME_WINDOWS_MAJOR_VER=84
CHROME_SUFFIX="_${CHROME_WINDOWS_MAJOR_VER}"


CHROME_DRIVER_VERSION=$(curl -sS https://chromedriver.storage.googleapis.com/LATEST_RELEASE${CHROME_SUFFIX})
CHROME_DRIVER_ARTIFACT=chromedriver_linux64.zip
CHROME_DRIVER_DEST=/usr/local/bin/chromedriver

# Clean workspace
rm ./$CHROME_DRIVER_ARTIFACT
sudo rm $CHROME_DRIVER_DEST

# Install dependencies
sudo apt-get install -y openjdk-8-jre-headless xvfb libxi6 libgconf-2-4

# Install chromedriver
wget -N https://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/$CHROME_DRIVER_ARTIFACT -P ./
unzip ./$CHROME_DRIVER_ARTIFACT -d ./
rm ./$CHROME_DRIVER_ARTIFACT
sudo mv -f ./chromedriver $CHROME_DRIVER_DEST
sudo chown root:root $CHROME_DRIVER_DEST
sudo chmod 0755 $CHROME_DRIVER_DEST
