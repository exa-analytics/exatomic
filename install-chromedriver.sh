#!/usr/bin/env bash

echo """
Installer script to set up chromedriver for selenium.
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

ON_WSL=1                     # 1 if on WSL
CHROME_WINDOWS_MAJOR_VER=84  # check your chrome version

# ======

CHROME_DRIVER_ARTIFACT=chromedriver_linux64.zip
CHROME_DRIVER_DEST=/usr/local/bin/chromedriver
CHROME_DEB=google-chrome-stable_current_amd64.deb

# Clean workspace
rm ./${CHROME_DRIVER_ARTIFACT}
sudo rm ${CHROME_DRIVER_DEST}
rm ./${CHROME_DEB}

# Install dependencies
sudo apt-get install -y openjdk-8-jre-headless xvfb libxi6 libgconf-2-4

if [[ "${ON_WSL}" == 0 ]]; then
    # Download chrome if normal linux
    wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
    wget https://dl.google.com/linux/direct/${CHROME_DEB}
    sudo dpkg -i ${CHROME_DEB}
    sudo apt -f install
    sudo dpkg -i ${CHROME_DEB}
    CHROME_DRIVER_VERSION=$(google-chrome-stable --version | cut -d ' ' -f 3)
else
    CHROME_SUFFIX="_${CHROME_WINDOWS_MAJOR_VER}"
    CHROME_DRIVER_VERSION=$(curl -sS https://chromedriver.storage.googleapis.com/LATEST_RELEASE${CHROME_SUFFIX})
fi

# Install chromedriver
wget -N https://chromedriver.storage.googleapis.com/${CHROME_DRIVER_VERSION}/${CHROME_DRIVER_ARTIFACT} -P ./
unzip ./${CHROME_DRIVER_ARTIFACT} -d ./
rm ./${CHROME_DRIVER_ARTIFACT}
sudo mv -f ./chromedriver ${CHROME_DRIVER_DEST}
sudo chown root:root ${CHROME_DRIVER_DEST}
sudo chmod 0755 ${CHROME_DRIVER_DEST}
