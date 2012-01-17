#!/bin/bash

i="0"

while [ $i -lt 5000 ]
do
head -n$i plot.tok.gt9.5000 | tail -n1 > objective/default/file$i
head -n$i quote.tok.gt9.5000 | tail -n1 > subjective/default/file$i
i=$[$i+1]
done