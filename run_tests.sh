#!/usr/bin/env bash
#
# run_tests.sh
#
# For each test in $TESTS (exactly once):
#  • Launch ROS in a new xterm
#  • Run SplaTAM (current terminal)
#  • After SplaTAM exits, kill the ROS xterm
#  • Move on to the next test
#

########### USER‐EDITABLE SECTION ###########

# Delay (in seconds) to allow ROS to spin up before we launch SplaTAM
ROS_STARTUP_DELAY=15

# Define one line per test. Each line = "<config>.py <flags…>"
# (Without “configs/habitat/” prefix, since the script adds that automatically.)
TESTS=(
    # Final Voxblox
    # "splatam.py --k_sil 0 --k_eig 0 --k_sum 0 --run_name final_vox1"
    # "splatam.py --k_sil 0 --k_eig 0 --k_sum 0 --run_name final_vox2"
    # "splatam.py --k_sil 0 --k_eig 0 --k_sum 0 --run_name final_vox3"

    # Final Silhouette
    # "splatam.py --k_sil 1500 --k_eig 0 --run_name final_sil1"
    # "splatam.py --k_sil 1500 --k_eig 0 --run_name final_sil2"
    # "splatam.py --k_sil 1500 --k_eig 0 --run_name final_sil3"

    # Final Fisher
    # "splatam.py --k_sil 1500 --k_eig 1 --run_name final_eig1"
    # "splatam.py --k_sil 1500 --k_eig 1 --run_name final_eig2"
    # "splatam.py --k_sil 1500 --k_eig 1 --run_name final_eig3"
    
    # "splatam.py --k_sil 1500 --k_eig 1 --nl_eig --run_name final_eig1_nl"
    # "splatam.py --k_sil 1500 --k_eig 1 --nl_eig --run_name final_eig2_nl"
    # "splatam.py --k_sil 1500 --k_eig 1 --nl_eig --run_name final_eig3_nl"

    "splatam.py --k_sil 1500 --k_eig 1 --k_sum 0.5 --nl_eig --run_name final_sum0"
    "splatam.py --k_sil 1500 --k_eig 1 --k_sum 0.5 --nl_eig --run_name final_sum1"
    #"splatam.py --k_sil 1500 --k_eig 1 --k_sum 0.5 --nl_eig --run_name final_sum"
    
    #without scaling
    #"splatam.py --k_sil 0 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name testVox --map_iter 30"


    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name trajFISH00 --map_iter 30"
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name trajFISH01 --map_iter 30"
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name trajFISH02 --map_iter 30"
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name trajFISH03 --map_iter 30"
    #"splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name trajFISH04 --map_iter 30"    
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name trajFISH01 --map_iter 30"       
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name trajFISH02 --map_iter 30"

    #test silhouette
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name trajSIL00 --map_iter 30"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name trajSIL01 --map_iter 30"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name trajSIL02 --map_iter 30"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name trajSIL03 --map_iter 30"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name trajSIL04 --map_iter 30"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name trajSIL05 --map_iter 30"
    
    # #non linear gains
    # "splatam.py --k_sil 0 --k_eig 1.5 --k_sum 1 --disable_monte --n_monte 0 --run_name fish10  --nl_eig  --map_iter 30"
    # "splatam.py --k_sil 0 --k_eig 1.5 --k_sum 1 --disable_monte --n_monte 0 --run_name fish11 --nl_eig  --map_iter 30"
    # "splatam.py --k_sil 0 --k_eig 1.5 --k_sum 1 --disable_monte --n_monte 0 --run_name fish12 --nl_eig  --map_iter 30"
    # #no monte
    # "splatam.py --k_sil 0 --k_eig 1.5 --k_sum 1 --disable_monte --n_monte 0 --run_name monte00 --map_iter 30"
    # "splatam.py --k_sil 0 --k_eig 1.5 --k_sum 1 --disable_monte --n_monte 0 --run_name monte01 --map_iter 30"
    # "splatam.py --k_sil 0 --k_eig 1.5 --k_sum 1 --disable_monte --n_monte 0 --run_name monte02 --map_iter 30"
    # #monte 25
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 25 --run_name monte10 --map_iter 30"
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 25 --run_name monte12 --map_iter 30"
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 25 --run_name monte12 --map_iter 30"
    # #monte 40
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 40 --run_name monte20 --map_iter 30"
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 40 --run_name monte21 --map_iter 30"
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 40 --run_name monte22 --map_iter 30"
    # #monte 55
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 55 --run_name monte30 --map_iter 30"
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 55 --run_name monte31 --map_iter 30"
    # "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 55 --run_name monte32 --map_iter 30"
    # #test silhouette
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name sil00 --map_iter 30"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name sil01 --map_iter 30"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name silh02 --map_iter 30"
    # # #non linear gains
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name sil10  --nl_sil --map_iter 30"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name sil11  --nl_sil  --map_iter 30"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name sil12  --nl_sil  --map_iter 30"
    # #test silhouette
    # "splatam.py --k_sil 1200 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name sil20 --map_iter 30"
    # "splatam.py --k_sil 1200 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name sil21 --map_iter 30"
    # "splatam.py --k_sil 1200 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name sil22 --map_iter 30"
    # #test on mappping iterations with silhouette
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name sil30 --map_iter 50"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name sil31 --map_iter 50"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name sil32 --map_iter 50"

    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1  --disable_monte --n_monte 0 --run_name mediathr025scale1i0sil --map_iter 30 --median_thr 0.25  --median_scale 1"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1  --disable_monte --n_monte 0 --run_name mediathr025scale1i1sil --map_iter 30 --median_thr 0.25  --median_scale 1"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1  --disable_monte --n_monte 0 --run_name mediathr025scale1i2sil --map_iter 30 --median_thr 0.25  --median_scale 1"

    #     # #no thresholds scale 1 1
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1  --disable_monte --n_monte 0 --run_name mediathr1scale1i0sil --map_iter 30 --median_thr 1  --median_scale 1"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1  --disable_monte --n_monte 0 --run_name mediathr1scale1i1sil --map_iter 30 --median_thr 1  --median_scale 1"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1  --disable_monte --n_monte 0 --run_name mediathr1scale1i2sil --map_iter 30 --median_thr 1  --median_scale 1"

    # #BATCH mapping thresholds tests
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale40i0 --map_iter 30 --median_thr 10  --median_scale 50"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale40i1 --map_iter 30 --median_thr 10  --median_scale 50"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale40map60i3 --map_iter 60 --median_thr 10  --median_scale 50"
    # #no thresholds  scale 20
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale20i0 --map_iter 30 --median_thr 10  --median_scale 20"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale20i1 --map_iter 30 --median_thr 10  --median_scale 20"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale20i2 --map_iter 30 --median_thr 10  --median_scale 20"
    # #no thresholds scale 8
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale8i0 --map_iter 30 --median_thr 10  --median_scale 8"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale8i1 --map_iter 30 --median_thr 10  --median_scale 8"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale8i2 --map_iter 30 --median_thr 10  --median_scale 8"
    # #no thresholds scale 3
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale3i0 --map_iter 30 --median_thr 10  --median_scale 3"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale3i1 --map_iter 30 --median_thr 10  --median_scale 3"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale3i2 --map_iter 30 --median_thr 10  --median_scale 3"
    # #no thresholds scale 1
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale1i0 --map_iter 30 --median_thr 10  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale1i1 --map_iter 30 --median_thr 10  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr10scale1i2 --map_iter 30 --median_thr 10  --median_scale 1"
    # # thresholds 1 scale 1
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr1scale1i0 --map_iter 30 --median_thr 1  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr1scale1i1 --map_iter 30 --median_thr 1  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr1scale1i2 --map_iter 30 --median_thr 1  --median_scale 1"
    # # thresholds 0.5 scale 1
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr05scale1i0 --map_iter 30 --median_thr 0.5  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr05scale1i1 --map_iter 30 --median_thr 0.5  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr05scale1i2 --map_iter 30 --median_thr 0.5  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr05scale1i2map60i5 --map_iter 60 --median_thr 0.5  --median_scale 1"

    # # thresholds 0.1 scale 0.25
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr025scale1i0 --map_iter 30 --median_thr 0.25  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr025scale1i1 --map_iter 30 --median_thr 0.25  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr025scale1i2 --map_iter 30 --median_thr 0.25  --median_scale 1"
    # # thresholds 0.1 scale 1
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr01scale1i0 --map_iter 30 --median_thr 0.1  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr01scale1i1 --map_iter 30 --median_thr 0.1  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr01scale1i2 --map_iter 30 --median_thr 0.1  --median_scale 1"
    # # thresholds 0.1 scale 0.05
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr005scale1i0 --map_iter 30 --median_thr 0.05  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr005scale1i1 --map_iter 30 --median_thr 0.05  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr005scale1i2 --map_iter 30 --median_thr 0.05  --median_scale 1"
    # #add gaussian always
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr00scale1map50 --map_iter 50 --median_thr 0  --median_scale 1"
    # # thresholds 0.1 scale 1 map 50
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr01scale1map60i2 --map_iter 60 --median_thr 0.1  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr01scale1map60i3 --map_iter 60 --median_thr 0.1  --median_scale 1"
    # "splatam.py --k_sil 0 --k_eig 1 --k_sum 1 --disable_monte --n_monte 0 --run_name mediathr005scale1map60i4 --map_iter 60 --median_thr 0.05  --median_scale 1"

)

# Paths to scripts & configs
SPLATAM_SCRIPT="scripts/splatam_realtime.py"
CONFIG_DIR="configs/habitat"

# ROS launch command (unchanged for every test)
ROS_CMD="bash -lc '
  source /home/dev/catkin_ws/devel/setup.bash && \
  roslaunch ifpp_application run_experiment_mav_sim.launch planner:=nbsplus
'"

#############################################

echo "========================================"
echo "Running ${#TESTS[@]} tests (each exactly once)."
echo "ROS command: $ROS_CMD"
echo "SplaTAM script: $SPLATAM_SCRIPT"
echo "Config folder: $CONFIG_DIR/"
echo "========================================"
echo

# Loop over each test (no inner repetition)
for idx in "${!TESTS[@]}"; do
    testnum=$(( idx + 1 ))
    config_and_flags="${TESTS[$idx]}"

    # Extract just the config filename (first token) and the rest as flags
    config_file=$(echo $config_and_flags | awk '{print $1}')
    extra_args=$(echo $config_and_flags | cut -d' ' -f2-)

    echo "========== Test #$testnum / ${#TESTS[@]}: config=$config_file =========="
    echo "Flags: $extra_args"
    echo

    #
    # 1) Launch ROS in a new xterm window (background)
    #
    xterm -hold -e "$ROS_CMD" &
    ROS_TERM_PID=$!
    echo "  → Launched ROS in xterm (PID=$ROS_TERM_PID). Waiting $ROS_STARTUP_DELAY s…"
    sleep $ROS_STARTUP_DELAY

    #
    # 2) Run SplaTAM in this (current) terminal
    #
    FULL_CMD="python3 $SPLATAM_SCRIPT $CONFIG_DIR/$config_and_flags"
    echo "  → Starting SplaTAM:"
    echo "    $FULL_CMD"
    eval $FULL_CMD
    STATUS=$?
    echo "  ← SplaTAM exited (status $STATUS)."

    #
    # 3) Kill the ROS xterm (if still alive)
    #
    if ps -p $ROS_TERM_PID > /dev/null 2>&1; then
        echo "  → Killing ROS xterm (PID=$ROS_TERM_PID)…"
        kill $ROS_TERM_PID
        sleep 1
    else
        echo "  → ROS xterm (PID=$ROS_TERM_PID) already terminated."
    fi

    echo
    echo "========== Completed Test #$testnum =========="
    echo
done

echo "All ${#TESTS[@]} tests completed."
exit 0