#!/usr/bin/env bash
export CCPI_BUILD_ARGS="-c conda-forge -c ccpi -c astra-toolbox"
bash <(curl -L https://raw.githubusercontent.com/vais-ral/CCPi-VirtualMachine/master/scripts/jenkins-build.sh)
