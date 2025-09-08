collab_ours:
	CUDA_VISIBLE_DEVICES=0 \
	python src/main.py \
	--gnn_type OURS \
	--depth 2 \
	--hidden 256 \
	--out_feat 128 \
	--batch_size 65536 \
	--epochs 200 \
	--lr 0.001 \
	--dataset ogbl-collab \
	--n_steps 10 \
	--runs 1 \
	--feat_weight 0.1 \
	--lap_weight 0.09 \
	--recon_weight 1.0 \
	--feat_weight2 0.01 