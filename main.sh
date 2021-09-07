echo "Start process"
for loop in $(seq 1 1 8)
do
    for coef in $(seq 1 1 20)
    do
        for gamma in $(seq 1 1 20)
        do
            for set_idx in $(seq 0 5 19)
                do
                next_idx_1=$((set_idx+1))
                next_idx_2=$((set_idx+2))
                next_idx_3=$((set_idx+3))
                next_idx_4=$((set_idx+4))
                python -u main.py --patch_size=12 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$set_idx --gap=5 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_1 --gap=5 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_2 --gap=5 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_3 --gap=5 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_4 --gap=5 --loop=$loop
                
            done
        done
    done
    rm -rf data/

    echo "Start process"

    for coef in $(seq 1 1 20)
    do
        for gamma in $(seq 1 1 20)
        do
            for set_idx in $(seq 0 5 19)
                do
                next_idx_1=$((set_idx+1))
                next_idx_2=$((set_idx+2))
                next_idx_3=$((set_idx+3))
                next_idx_4=$((set_idx+4))
                python -u main.py --patch_size=8 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$set_idx --gap=5 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_1 --gap=5 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_2 --gap=5 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_3 --gap=5 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_4 --gap=5 --loop=$loop
                
            done
        done
    done
    rm -rf data/

    echo "Start process"

    for coef in $(seq 1 1 20)
    do
        for gamma in $(seq 1 1 20)
        do
            for set_idx in $(seq 0 5 19)
                do
                next_idx_1=$((set_idx+1))
                next_idx_2=$((set_idx+2))
                next_idx_3=$((set_idx+3))
                next_idx_4=$((set_idx+4))
                python -u main.py --patch_size=12 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$set_idx --gap=5 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_1 --gap=5 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_2 --gap=5 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_3 --gap=5 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_4 --gap=5 --loop=$loop
                
            done
        done
    done
    rm -rf data/

    echo "Start process"

    for coef in $(seq 1 1 20)
    do
        for gamma in $(seq 1 1 20)
        do
            for set_idx in $(seq 0 5 19)
                do
                next_idx_1=$((set_idx+1))
                next_idx_2=$((set_idx+2))
                next_idx_3=$((set_idx+3))
                next_idx_4=$((set_idx+4))
                python -u main.py --patch_size=8 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$set_idx --gap=5 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_1 --gap=5 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_2 --gap=5 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_3 --gap=5 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_4 --gap=5 --loop=$loop
            done
        done
    done
    rm -rf data/

    for coef in $(seq 1 1 20)
    do
        for gamma in $(seq 1 1 20)
        do
            for set_idx in $(seq 0 5 19)
                do
                next_idx_1=$((set_idx+1))
                next_idx_2=$((set_idx+2))
                next_idx_3=$((set_idx+3))
                next_idx_4=$((set_idx+4))
                python -u main.py --patch_size=12 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$set_idx --gap=3 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_1 --gap=3 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_2 --gap=3 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_3 --gap=3 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_4 --gap=3 --loop=$loop
                
            done
        done
    done
    rm -rf data/

    echo "Start process"

    for coef in $(seq 1 1 20)
    do
        for gamma in $(seq 1 1 20)
        do
            for set_idx in $(seq 0 5 19)
                do
                next_idx_1=$((set_idx+1))
                next_idx_2=$((set_idx+2))
                next_idx_3=$((set_idx+3))
                next_idx_4=$((set_idx+4))
                python -u main.py --patch_size=8 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$set_idx --gap=3 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_1 --gap=3 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_2 --gap=3 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_3 --gap=3 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=20 --coef=$coef --gamma=$gamma --set_idx=$next_idx_4 --gap=3 --loop=$loop
                
            done
        done
    done
    rm -rf data/

    echo "Start process"

    for coef in $(seq 1 1 20)
    do
        for gamma in $(seq 1 1 20)
        do
            for set_idx in $(seq 0 5 19)
                do
                next_idx_1=$((set_idx+1))
                next_idx_2=$((set_idx+2))
                next_idx_3=$((set_idx+3))
                next_idx_4=$((set_idx+4))
                python -u main.py --patch_size=12 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$set_idx --gap=3 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_1 --gap=3 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_2 --gap=3 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_3 --gap=3 --loop=$loop &
                python -u main.py --patch_size=12 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_4 --gap=3 --loop=$loop
                
            done
        done
    done
    rm -rf data/

    echo "Start process"

    for coef in $(seq 1 1 20)
    do
        for gamma in $(seq 1 1 20)
        do
            for set_idx in $(seq 0 5 19)
                do
                next_idx_1=$((set_idx+1))
                next_idx_2=$((set_idx+2))
                next_idx_3=$((set_idx+3))
                next_idx_4=$((set_idx+4))
                python -u main.py --patch_size=8 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$set_idx --gap=3 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_1 --gap=3 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_2 --gap=3 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_3 --gap=3 --loop=$loop &
                python -u main.py --patch_size=8 --num_clusters=50 --coef=$coef --gamma=$gamma --set_idx=$next_idx_4 --gap=3 --loop=$loop
            done
        done
    done
    rm -rf data/
done


