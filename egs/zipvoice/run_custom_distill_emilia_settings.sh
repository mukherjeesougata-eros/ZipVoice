export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=5
stop_stage=5

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Train the ZipVoice-Distill model (first stage)"
      python3 -m zipvoice.bin.train_zipvoice_distill \
            --world-size 1 \
            --use-fp16 1 \
            --num-iters 60000 \
            --max-duration 500 \
            --base-lr 0.0005 \
            --tokenizer simple \
            --token-file data/tokens_custom.txt \
            --dataset custom \
            --train-manifest data/fbank/custom_cuts_train.jsonl.gz \
            --dev-manifest data/fbank/custom_cuts_dev.jsonl.gz \
            --teacher-model exp/zipvoice_custom/default_setup/iter-60000-avg-2.pt \
            --distill-stage first \
            --exp-dir exp/zipvoice_distill_1stage_custom_emilia_setup \
            --start-epoch 2
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Average the checkpoints for ZipVoice-Distill (first stage)"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 60000 \
            --avg 7 \
            --model-name zipvoice_distill \
            --exp-dir exp/zipvoice_distill_1stage_custom_emilia_setup
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Train the ZipVoice-Distill model (second stage)"

      python3 -m zipvoice.bin.train_zipvoice_distill \
            --world-size 1 \
            --use-fp16 1 \
            --num-iters 2000 \
            --save-every-n 1000 \
            --max-duration 500 \
            --base-lr 0.0001 \
            --model-config conf/zipvoice_base.json \
            --tokenizer simple \
            --token-file data/tokens_custom.txt \
            --dataset custom \
            --train-manifest data/fbank/custom_cuts_train.jsonl.gz \
            --dev-manifest data/fbank/custom_cuts_dev.jsonl.gz \
            --teacher-model exp/zipvoice_distill_1stage_custom_emilia_setup/iter-60000-avg-7.pt \
            --distill-stage second \
            --exp-dir exp/zipvoice_distill_custom_emilia_setup
fi




### Export ONNX model

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Export ZipVoice-Distill ONNX model"
      python3 -m zipvoice.bin.onnx_export \
            --model-name zipvoice_distill \
            --model-dir exp/zipvoice_distill_custom_emilia_setup/ \
            --checkpoint-name checkpoint-2000.pt \
            --onnx-model-dir exp/zipvoice_distill_custom_emilia_setup/
fi

### Inference with PyTorch and ONNX models

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Inference of trun_custom_distill_emilia_settings.shhe ZipVoice-Distill model"
      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice_distill \
            --model-dir exp/zipvoice_distill_custom_emilia_setup/ \
            --checkpoint-name checkpoint-2000.pt \
            --tokenizer simple \
            --test-list data/Test/test_all_lang_renamed.tsv \
            --res-dir results/test_distill_custom_emilia_setup/wo_IndicTTSPunjabi/ \
            --num-step 8 \
            --guidance-scale 3 \
            --raw-evaluation True
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Inference with ZipVoic-Distill ONNX model"
      python3 -m zipvoice.bin.infer_zipvoice_onnx \
            --model-name zipvoice_distill \
            --onnx-int8 False \
            --model-dir exp/zipvoice_distill_custom_emilia_setup/ \
            --tokenizer simple \
            --test-list test.tsv \
            --res-dir results/test_distill_custom_emilia_setup_onnx
fi