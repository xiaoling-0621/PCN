model=PCN
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id 'mk96_tk5_dm64' \
  --model $model \
  --data custom \
  --data_name electricity \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --d_model 64 \
  --batch_size 32 \
  --itr 3 \
  --max_k 96\
  --top_k 5


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id 'mk96_tk5_dm64' \
  --model $model \
  --data custom \
  --data_name electricity \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --d_model 64 \
  --batch_size 32 \
  --itr 3 \
  --max_k 96\
  --top_k 5

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id 'mk96_tk5_dm64' \
  --model $model \
  --data custom \
  --data_name electricity \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --d_model 64 \
  --batch_size 32 \
  --itr 3 \
  --max_k 96\
  --top_k 5


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id 'mk96_tk5_dm64' \
  --model $model \
  --data custom \
  --data_name electricity \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --d_model 64 \
  --batch_size 32 \
  --itr 3 \
  --max_k 96\
  --top_k 5
