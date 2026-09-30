@echo off
cd /d D:\WorkSpace\MIE-MLPKT\model\MLKT4
D:/Anaconda3/python.exe -u train_kc_mastery.py --data_dir Code-S --emb_size 64 --batch_size 12 --max_iter 60
pause
