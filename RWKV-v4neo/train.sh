epoch=0

python train.py \
    --my_testing "r2r3r4" \
    --load_model "/mnt/kqwyf/models/rwkv/RWKV-5-World-3B-v2-20231113-ctx4096.pth" \
    --proj_dir "exp/genshin_finetune" \
    --data_file "/mnt/kqwyf/datasets/genshin/genshin_talks.pkl" \
    --data_type pickle_traces \
    --vocab_size 65536 \
    --ctx_len 4096 \
    --accumulate_grad_batches 8 \
    --epoch_steps 100 \
    --epoch_count 3 \
    --epoch_begin ${epoch} \
    --epoch_save 1 \
    --micro_bsz 1 \
    --n_layer 32 \
    --n_embd 2560 \
    --pre_ffn 0 \
    --head_qk 0 \
    --lr_init 1e-5 \
    --lr_final 1e-6 \
    --warmup_steps 50 \
    --beta1 0.9 \
    --beta2 0.999 \
    --adam_eps 1e-8 \
    --accelerator gpu \
    --devices 1 \
    --precision bf16 \
    --grad_cp 2
    #--strategy auto \
