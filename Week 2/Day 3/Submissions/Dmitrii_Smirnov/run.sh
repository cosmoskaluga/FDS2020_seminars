#!/bin/bash
time=$(date +%s)

GREEN='\033[0;32m'
WHITE='\033[1;37m'
NC='\033[0m'
echo -e "${GREEN}Foundation of Data Science course${NC}"
echo " "

#1. Download dataset
echo -e "${WHITE}Run${NC}"
python3 main.py


echo "Finished"
echo "CPU time: $(($(date +%s)-$time))s"
