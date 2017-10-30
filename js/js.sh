#!/bin/bash
npm install
jupyter nbextension install --py --sys-prefix exatomic
jupyter nbextension enable --py --sys-prefix exatomic
