for f in $(ls /links/groups/borgwardt/Data/DRIAMS_strat/DRIAMS-A/binned_6000/2018/); do
    echo ${f}
    python plot_duplicates.py --INPUT /links/groups/borgwardt/Data/DRIAMS_strat/DRIAMS-A/binned_6000/2018/${f}/*
done
