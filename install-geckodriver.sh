#!/usr/bin/env bash


echo """
Debian-based installer script for the GeckoDriver Firefox selenium driver tool.
"""

GECKO_DRIVER_VERSION=v0.27.0
GECKO_DRIVER_BASE_URL=https://github.com/mozilla/geckodriver/releases/download
GECKO_DRIVER_ARTIFACT=geckodriver-${GECKO_DRIVER_VERSION}-linux64.tar.gz
GECKO_DRIVER=geckodriver
GECKO_DRIVER_SRC=${GECKO_DRIVER}.tar.gz 
GECKO_DRIVER_DEST=/usr/local/bin
GECKO_DRIVER_URL=${GECKO_DRIVER_BASE_URL}/${GECKO_DRIVER_VERSION}/${GECKO_DRIVER_ARTIFACT}

# Clean workspace
rm -f ./${GECKO_DRIVER_ARTIFACT}
rm -f ${GECKO_DRIVER_DEST}
rm -f ./${GECKO_DRIVER_SRC}

wget ${GECKO_DRIVER_URL} -O ${GECKO_DRIVER_SRC}
tar -xzvf ${GECKO_DRIVER_SRC}
rm -f ./${GECKO_DRIVER}.log
sudo mv -f ./${GECKO_DRIVER} ${GECKO_DRIVER_DEST}
rm ./${GECKO_DRIVER_SRC}
sudo chown root:root ${GECKO_DRIVER_DEST}/${GECKO_DRIVER}
sudo chmod 0755 ${GECKO_DRIVER_DEST}/${GECKO_DRIVER}
