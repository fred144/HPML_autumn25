#!/bin/bash
{ ./c1
echo ""
    ./c2
echo ""
    ./c3
} | tee -a STDOUT