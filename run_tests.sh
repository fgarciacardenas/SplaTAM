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
    #without scaling
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish0_0 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish0_1 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish0_2 --map_iter 30"
    #non linear gains
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish1_0 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish1_1 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish1_2 --map_iter 30"
    #no monte
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name monte0_0 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name monte0_1 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1 --disable_monte --n_monte 0 --run_name monte0_2 --map_iter 30"
    #monte 25
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 25 --run_name monte1_0 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 25 --run_name monte1_1 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 25 --run_name monte1_2 --map_iter 30"
    #monte 40
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 40 --run_name monte2_0 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 40 --run_name monte2_1 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 40 --run_name monte2_2 --map_iter 30"
    #monte 55
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 55 --run_name monte3_0 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 55 --run_name monte3_1 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 55 --run_name monte3_2 --map_iter 30"
    #monte 70
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 70 --run_name monte4_0 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 70 --run_name monte4_1 --map_iter 30"
    "splatam.py --k_sil 0 --k_eig 1.0 --k_sum 1  --n_monte 70 --run_name monte4_2 --map_iter 30"
    #test silhouette
    "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish0_0 --map_iter 30"
    "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish0_1 --map_iter 30"
    "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish0_2 --map_iter 30"
    # #non linear gains
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish1_0 --map_iter 30"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish1_1 --map_iter 30"
    # "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish1_2 --map_iter 30"
    #test silhouette
    "splatam.py --k_sil 1200 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish0_0 --map_iter 30"
    "splatam.py --k_sil 1200 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish0_1 --map_iter 30"
    "splatam.py --k_sil 1200 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish0_2 --map_iter 30"
    #test on mappping iterations with silhouette
    "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish0_0 --map_iter 50"
    "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish0_1 --map_iter 50"
    "splatam.py --k_sil 1500 --k_eig 0 --k_sum 1 --disable_monte --n_monte 0 --run_name fish0_2 --map_iter 50"

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