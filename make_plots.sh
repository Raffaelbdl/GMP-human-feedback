bash .env/bin/activate

python train.py --architecture style --m_n_layers 0 --task cartpole
python train.py --architecture style --m_n_layers 4 --task cartpole
python train.py --architecture style --m_n_layers 8 --task cartpole
python train.py --architecture multiplicative --m_n_layers 0 --task cartpole
python train.py --architecture multiplicative --m_n_layers 4 --task cartpole
python train.py --architecture multiplicative --m_n_layers 8 --task cartpole

python train.py --architecture style --m_n_layers 0 --task ring
python train.py --architecture style --m_n_layers 4 --task ring
python train.py --architecture style --m_n_layers 8 --task ring
python train.py --architecture multiplicative --m_n_layers 4 --task ring
python train.py --architecture multiplicative --m_n_layers 8 --task ring
python train.py --architecture multiplicative --m_n_layers 0 --task ring