CUDA_VISIBLE_DEVICES=7 python eval.py \
  --model-path /home/sensing_test/Jiang_Zheng/models/Qwen3-VL-8B-Instruct \
  --pairs-file pairs.csv \
  --epochs 10 \
  --output result.csv

# csv转execl表格：
python csv2xlsx.py -i result.csv -o result.xlsx --autowidth


modelscope download --model 'Qwen/Qwen3-VL-30B-A3B-Instruct' --local_dir 'Qwen3-VL-30B-A3B-Instruct'

# csv转execl表格新：
python csv2xlsx2.py --input-dir csv_output --output-dir ./xlsx_output --autowidth