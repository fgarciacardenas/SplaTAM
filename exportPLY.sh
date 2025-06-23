#!/usr/bin/env bash
#
# exprotPLY.sh
#

TESTS=(
    # #without scaling
    # "splatam.py  --run_name trajFISH00 "
    # "splatam.py --run_name trajFISH01"
    # "splatam.py --run_name trajFISH02 "
    # "splatam.py  --run_name trajFISH03"
    # "splatam.py --run_name trajFISH04 "    
    # "splatam.py  --run_name trajFISH01 "       
    # "splatam.py --run_name trajFISH02 "

    # #test silhouette
    # "splatam.py  --run_name trajSIL00 "
    # "splatam.py --run_name trajSIL01 "
    # "splatam.py --run_name trajSIL02 "
    # "splatam.py --run_name trajSIL03 "
    # "splatam.py --run_name trajSIL04 "
    # "splatam.py  --run_name trajSIL05 "
    
    # #non linear gains
    # "splatam.py --run_name fish10  --nl_eig "
    # "splatam.py --run_name fish11 --nl_eig  "
    # "splatam.py --run_name fish12 --nl_eig "
    # #no monte
    # "splatam.py --run_name monte00 "
    # "splatam.py  --run_name monte01 "
    # "splatam.py --run_name monte02 "
    # #monte 25
    # "splatam.py --run_name monte10 "
    # "splatam.py -run_name monte12 "
    # "splatam.py --run_name monte12 "
    # #monte 40
    # "splatam.py --run_name monte20 "
    # "splatam.py --run_name monte21 "
    # "splatam.py  --run_name monte22 "
    # #monte 55
    # "splatam.py --run_name monte30 "
    # "splatam.py  --run_name monte31 "
    # "splatam.py --run_name monte32 "
    # #test silhouette
    # "splatam.py --run_name sil00 "
    # "splatam.py  --run_name sil01 "
    # "splatam.py --run_name silh02 "
    # # #non linear gains
    # "splatam.py  --run_name sil10  "
    # "splatam.py  --run_name sil11 "
    # "splatam.py --run_name sil12 "
    # #test silhouette
    # "splatam.py  --run_name sil20 "
    # "splatam.py --run_name sil21 "
    # "splatam.py  --run_name sil22 "
    # #test on mappping iterations with silhouette
    # "splatam.py --run_name sil30 "
    # "splatam.py  --run_name sil31 "
    # "splatam.py --run_name sil32 "

    # "splatam.py  --run_name mediathr025scale1i0sil"
    # "splatam.py  --run_name mediathr025scale1i1sil "
    # "splatam.py  --run_name mediathr025scale1i2sil"

    #     # #no thresholds scale 1 1
    # "splatam.py --run_name mediathr1scale1i0sil "
    # "splatam.py --run_name mediathr1scale1i1sil "
    # "splatam.py --run_name mediathr1scale1i2sil "

    # #BATCH mapping thresholds tests
    # "splatam.py --run_name mediathr10scale40i0 "
    # "splatam.py --run_name mediathr10scale40i1 "
    # "splatam.py --run_name mediathr10scale40map60i3 "
    # #no thresholds  scale 20
    # "splatam.py --run_name mediathr10scale20i0 "
    # "splatam.py --run_name mediathr10scale20i1"
    # "splatam.py --run_name mediathr10scale20i2"
    # #no thresholds scale 8
    # "splatam.py --run_name mediathr10scale8i0 "
    # "splatam.py --run_name mediathr10scale8i1 "
    # "splatam.py --run_name mediathr10scale8i2"
    # #no thresholds scale 3
    # "splatam.py  --run_name mediathr10scale3i0 "
    # "splatam.py  --run_name mediathr10scale3i1 "
    # "splatam.py --run_name mediathr10scale3i2 "
    # #no thresholds scale 1
    # "splatam.py  --run_name mediathr10scale1i0 "
    # "splatam.py  --run_name mediathr10scale1i1 "
    # "splatam.py  --run_name mediathr10scale1i2 "
    # # thresholds 1 scale 1
    # "splatam.py  --run_name mediathr1scale1i0 "
    # "splatam.py  --run_name mediathr1scale1i1 "
    # "splatam.py --run_name mediathr1scale1i2 "
    # # thresholds 0.5 scale 1
    # "splatam.py --run_name mediathr05scale1i0 "
    # "splatam.py --run_name mediathr05scale1i1 "
    # "splatam.py --run_name mediathr05scale1i2 "
    # "splatam.py  --run_name mediathr05scale1i2map60i5 "

    # # thresholds 0.1 scale 0.25
    # "splatam.py --run_name mediathr025scale1i0"
    # "splatam.py --run_name mediathr025scale1i1 "
    # "splatam.py --run_name mediathr025scale1i2 "
    # # thresholds 0.1 scale 1
    # "splatam.py  --run_name mediathr01scale1i0 "
    # "splatam.py  --run_name mediathr01scale1i1 "
    # "splatam.py  --run_name mediathr01scale1i2 "
    # # thresholds 0.1 scale 0.05
    # "splatam.py  --run_name mediathr005scale1i0 "
    # "splatam.py --run_name mediathr005scale1i1 "
    # "splatam.py  --run_name mediathr005scale1i2 "
    # #add gaussian always
    # "splatam.py --run_name mediathr00scale1map50"
    # # thresholds 0.1 scale 1 map 50
    # "splatam.py --run_name mediathr01scale1map60i2 "
    # "splatam.py --run_name mediathr01scale1map60i3 "
    # "splatam.py --run_name mediathr005scale1map60i4 "

    # Final Voxblox
    "splatam.py  --run_name final_vox1"
    "splatam.py --run_name final_vox2"
    "splatam.py  --run_name final_vox3"

    # Final Silhouette
    "splatam.py  --run_name final_sil1"
    "splatam.py  --run_name final_sil2"
    "splatam.py  --run_name final_sil3"

    # Final Fisher
    
    "splatam.py  --run_name final_eig1_nl"
    "splatam.py  --run_name final_eig2_nl"
    "splatam.py --run_name final_eig3_nl"
    #final combo
    "splatam.py  --run_name final_sum0"
    "splatam.py --run_name final_sum1"
    "splatam.py  --run_name final_sum"
)

# Paths to your script & config folder
SPLATAM_SCRIPT="scripts/export_ply.py"
CONFIG_DIR="configs/habitat"

#############################################

echo "========================================"
echo "Running ${#TESTS[@]} tests."
echo "SplaTAM script: $SPLATAM_SCRIPT"
echo "Config directory: $CONFIG_DIR/"
echo "========================================"
echo

for idx in "${!TESTS[@]}"; do
  testnum=$(( idx + 1 ))
  config_and_flags="${TESTS[$idx]}"

  # Split out the config filename (first token) and flags (rest)
  config_file=$(awk '{print $1}' <<<"$config_and_flags")
  extra_args=$(cut -d' ' -f2- <<<"$config_and_flags")

  echo "---- Test #$testnum/${#TESTS[@]}: $config_file ----"
  echo "Flags: $extra_args"
  echo

  FULL_CMD="python3 $SPLATAM_SCRIPT $CONFIG_DIR/$config_file $extra_args"
  echo "→ Running: $FULL_CMD"
  eval $FULL_CMD
  status=$?
  echo "← Finished (exit code $status)"
  echo
done

echo "All ${#TESTS[@]} tests completed."
exit 0