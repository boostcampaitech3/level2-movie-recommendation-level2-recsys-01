# python DeepCTR.py --model DeepFM --embedding_dim 10 --dropout 0.5 --activation gelu --batch_size 128 --num_epochs 50
# python DeepCTR.py --model AutoInt --embedding_dim 10 --dropout 0.5 --activation gelu --batch_size 128 --num_epochs 50
# python DeepCTR.py --model DCNMix --embedding_dim 10 --dropout 0.5 --activation gelu --batch_size 128 --num_epochs 50
# python DeepCTR.py --model AutoInt --embedding_dim 64 --dropout 0.5 --activation gelu --batch_size 256 --num_epochs 500
python DeepCTR.py --model AutoInt --embedding_dim 64 --dropout 0.5 --activation gelu --batch_size 256 --num_epochs 5