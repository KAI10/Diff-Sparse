#!/bin/bash

patchSize=80
numPatches=10
ECH=1
COV=1
MASKP=0.9

storePathPrefix=output/mask_"$MASKP"

for i in {0..9};
do
  echo "Submitting Job with: patchSize $patchSize numPatches $numPatches ECH $ECH, COV $COV, MASKP $MASKP Seed $i ..."

  mkdir -p "$storePathPrefix"/$patchSize/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"
  sbatch --export=seed="$i",storePath="$storePathPrefix"/$patchSize/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"/ \
          --job-name=DF-"$patchSize"-"$numPatches"_ECH_"$ECH"_COV_"$COV"_MASKP_"$MASKP"-"$i" \
          --error="$storePathPrefix"/"$patchSize"/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"/train.err \
          --output="$storePathPrefix"/"$patchSize"/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"/train.out \
          worker.sh > "$storePathPrefix"/"$patchSize"/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"/train.jobid

  sleep 3s
done
