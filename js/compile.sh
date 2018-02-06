#!/usr/bin/env bash


for path in `ls src/*.ts`; do
    echo -e "\nCompiling TypeScript: $path";
    tsc $path;
done

echo -e "\nInstalling JavaScript"
npm install
