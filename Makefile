VIDEOS ?= /home/rafi/Downloads/blackrover_hw/videos

.PHONY: reproduce reproduce-stage1 evaluate ablation clean-cache

reproduce:
	python run.py --dataset-dir $(VIDEOS) --overwrite-loading --overwrite-algo \
	  --tracker-type botsort --reid-backbone ensemble \
	  --use-new-clustering --scene-backend gemini --evaluate

reproduce-stage1:
	python run.py --dataset-dir $(VIDEOS) --overwrite-algo \
	  --tracker-type botsort --use-rerank --cooccurrence-constraint --legacy-clustering

evaluate:
	python evaluate.py

ablation:
	python scripts/ablation.py --dataset-dir $(VIDEOS)

clean-cache:
	rm -rf $(VIDEOS)/.cache
