#!/bin/bash

qpdf --empty --pages head.pdf body.pdf -- combined.pdf

mv combined.pdf 'Daniel Kuknyo - Traffic Control and Infrastructure Organization Using Reinforcement Learning.pdf'

echo "done."
