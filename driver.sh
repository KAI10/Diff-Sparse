#!/bin/bash

patchSize=64
numPatches=10
ECH=1
COV=1
MASKP=0.95
CLOSS=0

# contextLength=12
# horizonLength=60

storePathPrefix=output/mask_"$MASKP"
# storePathPrefix=output/zero_mask/mask_"$MASKP"/
# storePathPrefix=output/mask_"$MASKP"/CLOSS_"$CLOSS"

for i in {0..9};
do
  echo "Submitting Job with: patchSize $patchSize numPatches $numPatches ECH $ECH, COV $COV, MASKP $MASKP Seed $i ..."

  mkdir -p "$storePathPrefix"/$patchSize/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"
  sbatch --export=seed="$i",storePath="$storePathPrefix"/$patchSize/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"/ \
          --job-name=DF-"$patchSize"-"$numPatches"_ECH_"$ECH"_COV_"$COV"_MASKP_"$MASKP"-"$i" \
          --error="$storePathPrefix"/"$patchSize"/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"/train.err \
          --output="$storePathPrefix"/"$patchSize"/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"/train.out \
          worker.sh > "$storePathPrefix"/"$patchSize"/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"/train.jobid

  # sbatch --export=seed="$i",storePath="$storePathPrefix"/$patchSize/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"/ \
  #         --job-name=DF-Test-"$patchSize"-"$numPatches"_ECH_"$ECH"_COV_"$COV"_MASKP_"$MASKP"_HORIZON_"$horizonLength"-"$i" \
  #         --error="$storePathPrefix"/"$patchSize"/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"/test_"$horizonLength".err \
  #         --output="$storePathPrefix"/"$patchSize"/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"/test_"$horizonLength".out \
  #         worker.sh > "$storePathPrefix"/"$patchSize"/num_patches_"$numPatches"_ECH_"$ECH"_COV_"$COV"/seed_"$i"/test_"$horizonLength".jobid

  sleep 3s
done
