export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=4

#### (Optional) Training ZipVoice-Distill model (4 - 7)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 4: Train the ZipVoice-Distill model (first stage)"
      python3 -m zipvoice.bin.train_zipvoice_distill_custom \
            --world-size 4 \
            --use-fp16 1 \
            --num-epochs 6 \
            --max-duration 500 \
            --base-lr 0.001 \
            --max-len 20.5 \
            --model-config conf/zipvoice_base.json \
            --tokenizer simple \
            --token-file data/tokens_custom.txt \
            --dataset custom \
            --train-manifest data/fbank/custom_cuts_train.jsonl.gz \
            --dev-manifest data/fbank/custom_cuts_dev.jsonl.gz \
            --teacher-model exp/zipvoice_custom/default_setup/iter-60000-avg-2.pt \
            --distill-stage "first" \
            --exp-dir exp/zipvoice_distill_1stage_custom_libritts_setup_rough
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 5: Average the checkpoints for ZipVoice-Distill (first stage)"
      python3 -m zipvoice.bin.generate_averaged_model \
            --epoch 6 \
            --avg 3 \
            --model-name zipvoice_distill \
            --exp-dir exp/zipvoice_distill_1stage_custom_libritts_setup_rough
      # The generated model is exp/zipvoice_distill_1stage_custom_libritts_setup/epoch-6-avg-3.pt
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 6: Train the ZipVoice-Distill model (second stage)"

      python3 -m zipvoice.bin.train_zipvoice_distill \
            --world-size 4 \
            --use-fp16 1 \
            --num-epochs 6 \
            --max-duration 500 \
            --base-lr 0.001 \
            --max-len 20.5 \
            --model-config conf/zipvoice_base.json \
            --tokenizer simple \
            --token-file data/tokens_custom.txt \
            --dataset custom \
            --train-manifest data/fbank/custom_cuts_train.jsonl.gz \
            --dev-manifest data/fbank/custom_cuts_dev.jsonl.gz \
            --teacher-model exp/zipvoice_distill_1stage_custom/epoch-6-avg-3.pt \
            --distill-stage second \
            --exp-dir exp/zipvoice_distill_custom_libritts_setup_rough
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 7: Average the checkpoints for ZipVoice-Distill (second stage)"
      python3 -m zipvoice.bin.generate_averaged_model \
            --epoch 6 \
            --avg 3 \
            --model-name zipvoice_distill \
            --exp-dir exp/zipvoice_distill_custom_libritts_setup_rough
      # The generated model is exp/zipvoice_distill_custom_libritts_setup/epoch-6-avg-3.pt
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 9: Inference of the ZipVoice-Distill model"
      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice_distill \
            --model-dir exp/zipvoice_distill_custom_libritts_setup \
            --checkpoint-name epoch-6-avg-3.pt \
            --tokenizer simple \
            --test-list test.tsv \
            --res-dir results/test_distill_custom_libritts_setup \
            --num-step 4 \
            --guidance-scale 3 \
            --t-shift 0.7 \
            --raw-evaluation True
fi