#!/usr/bin/env bash

echo """
Debian-based installer script to set up chromedriver for selenium.
First installs some linux dependencies for this script.
Then installs google-chrome-stable from google repository.
Uses that version of chrome to install chromedriver into PATH.
This should work for linux systems to run headless selenium
tests against chrome.

On WSL, however, installing chrome this way does not work in
headless mode. You can still run the selenium tests by pointing
to the chrome executable path from Windows.

Note:
    WSL environment should look something like:

    export DISPLAY=:0
    export BROWSER=/mnt/c/Program\ Files\ \(x86\)/Google/Chrome/Application/chrome.exe
"""

ON_WSL=0                     # 1 if on WSL
CHROME_WINDOWS_MAJOR_VER=84  # check your chrome version

# ======

CHROME_PUBKEY_URL=https://dl.google.com/linux/linux_signing_key.pub
CHROME_STABLE_URL=https://dl.google.com/linux/direct
CHROME_STABLE_DEB=google-chrome-stable_current_amd64.deb
CHROME_DRIVER_URL=https://chromedriver.storage.googleapis.com
CHROME_DRIVER_DEST=/usr/local/bin/chromedriver
CHROME_DRIVER_ARTIFACT=chromedriver_linux64.zip

# Clean workspace
rm -f ./${CHROME_DRIVER_ARTIFACT}
sudo rm -f ${CHROME_DRIVER_DEST}
rm -f ./${CHROME_DEB}

# Install dependencies
sudo apt-get install -y openjdk-8-jre-headless xvfb libxi6 libgconf-2-4

if [[ "${ON_WSL}" == 0 ]]; then
    # Download chrome if normal linux
    wget -q -O - ${CHROME_PUBKEY_URL} | sudo apt-key add -
    wget ${CHROME_STABLE_URL}/${CHROME_DEB}
    sudo dpkg -i ${CHROME_DEB}
    sudo apt -f install
    sudo dpkg -i ${CHROME_DEB}
    CHROME_SUFFIX=$(google-chrome-stable --version | cut -d ' ' -f 3 | cut -d '.' -f 1)
else
    CHROME_SUFFIX="${CHROME_WINDOWS_MAJOR_VER}"
fi
CHROME_DRIVER_VERSION=$(curl -sS ${CHROME_DRIVER_URL}/LATEST_RELEASE_${CHROME_SUFFIX})

# Install chromedriver
wget -N ${CHROME_DRIVER_URL}/${CHROME_DRIVER_VERSION}/${CHROME_DRIVER_ARTIFACT} -P ./
unzip ./${CHROME_DRIVER_ARTIFACT} -d ./
rm ./${CHROME_DRIVER_ARTIFACT}
sudo mv -f ./chromedriver ${CHROME_DRIVER_DEST}
sudo chown root:root ${CHROME_DRIVER_DEST}
sudo chmod 0755 ${CHROME_DRIVER_DEST}
