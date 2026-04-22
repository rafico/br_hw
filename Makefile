VIDEOS ?= /home/rafi/Downloads/blackrover_hw/videos
PYTHON ?= python3

.PHONY: reproduce reproduce-stage1 reproduce-ft reproduce-rerun evaluate ablation clean-cache qa-unit qa-offline qa-dataset-smoke qa-release-local qa-manual-visual-prep qa-validate

reproduce:
	$(PYTHON) run.py --dataset-dir $(VIDEOS) --overwrite-loading --overwrite-algo \
	  --tracker-type botsort --reid-backbone ensemble \
	  --use-new-clustering --scene-backend gemini --evaluate

reproduce-ft:
	$(PYTHON) run.py --dataset-dir $(VIDEOS) --overwrite-loading --overwrite-algo \
	  --tracker-type botsort --reid-backbone ensemble \
	  --use-new-clustering --scene-backend gemini --finetune-reid --evaluate

reproduce-rerun:
	$(PYTHON) run.py --dataset-dir $(VIDEOS) --overwrite-loading --overwrite-algo \
	  --tracker-type botsort --reid-backbone ensemble \
	  --use-new-clustering --scene-backend gemini \
	  --visualizer rerun --rerun-save recording.rrd

reproduce-stage1:
	$(PYTHON) run.py --dataset-dir $(VIDEOS) --overwrite-algo \
	  --tracker-type botsort --use-rerank --cooccurrence-constraint --legacy-clustering

evaluate:
	$(PYTHON) evaluate.py

ablation:
	$(PYTHON) scripts/ablation.py --dataset-dir $(VIDEOS)

qa-unit:
	$(PYTHON) scripts/run_qa.py --suite unit --python $(PYTHON)

qa-offline:
	$(PYTHON) scripts/run_qa.py --suite offline --python $(PYTHON)

qa-dataset-smoke:
	$(PYTHON) scripts/run_qa.py --suite dataset-smoke --python $(PYTHON) --dataset-dir $(VIDEOS)

qa-release-local:
	$(PYTHON) scripts/run_qa.py --suite release-local --python $(PYTHON) --dataset-dir $(VIDEOS)

qa-manual-visual-prep:
	$(PYTHON) scripts/run_qa.py --suite manual-visual-prep --python $(PYTHON) --dataset-dir $(VIDEOS) --rerun-save qa_artifacts/recording.rrd

qa-validate:
	$(PYTHON) scripts/qa_validate_outputs.py --catalogue catalogue_v2.json --scene scene_labels_v2.json --eval-report eval_report.json

clean-cache:
	rm -rf $(VIDEOS)/.cache
