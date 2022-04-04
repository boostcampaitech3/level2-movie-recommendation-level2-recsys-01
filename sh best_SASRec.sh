cd input
cd code
python preprocessing.py --attribute_name genre 
python run_pretrain.py --hidden_size 128 --num_hidden_layers 3 --num_attention_heads 4 --attribute_name genre_hidden_size_128_hidden_layers_3_attention_heads_4 
python run_train.py --using_pretrain --hidden_size 128 --num_hidden_layers 3 --num_attention_heads 4 --attribute_name genre_hidden_size_128_hidden_layers_3_attention_heads_4 
python inference.py --hidden_size 128 --num_hidden_layers 3 --num_attention_heads 4 --attribute_name genre_hidden_size_128_hidden_layers_3_attention_heads_4