root=/mnt/HDD1/phudh/phyai/data/copy_from_94/checkpoint_itri/checkpoint_itri

find "$root" -name "latest.ckpt" -type f | while read -r ckpt_path; do
    echo "Processing: $ckpt_path"
    uv run scripts/khanh_eval_offline.py \
      --checkpoint "$ckpt_path" \
      --split /mnt/HDD1/phudh/phyai/data/umi_episode_split.json \
      --num_workers 16 --batch_size 1
done