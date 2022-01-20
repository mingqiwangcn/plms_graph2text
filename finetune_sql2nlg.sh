#!/bin/bash

if [ "$#" -lt 2 ]; then
  echo "./finetune_sql2nlg.sh <model> <gpu_id>"
  exit 2
fi

if [[ ${1} == *"t5"* ]]; then
  bash sql2nlg/finetune_graph2text.sh ${1} ${2}
fi
if [[ ${1} == *"bart"* ]]; then
  bash sql2nlg/finetune_graph2text_bart.sh ${1} ${2}
fi








