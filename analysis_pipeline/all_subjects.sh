set -e

subjects=(
    sub-I0002150000008
    sub-I0002150000012
    sub-I0002150000013
    sub-I0002150000014
    sub-I0002150000031
)
 
for sub in "${subjects[@]}"; do
    echo ""
    echo "=================================================="
    echo "Processing $sub"
    echo "=================================================="
    export MROP_SUBJECT="$sub"
 
    echo "-- 2_preprocessing.py --"
    python 2_preprocessing.py
 
    echo "-- 3_extract_windows.py --"
    python 3_extract_windows.py
 
    echo "-- 4_analyze_compare.py --"
    python 4_analyze_compare.py
 
    echo "Done with $sub"
done
 
echo ""
