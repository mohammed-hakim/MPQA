# Running on Kaggle

**Easiest: use the notebook** `notebooks/kaggle_run.ipynb`. Upload it to Kaggle (File -> Import notebook), set the
accelerator to **GPU T4 x2** and Internet to **on**, add the dataset, edit the first code cell (`REPO_URL`, `DATA_DIR`,
`EPOCHS`) and run the cells in order. Everything below is what the notebook does, for reference.

1. New notebook: Settings -> Accelerator **GPU T4 x2**, **Internet on** (timm downloads the pretrained weights),
   add the MRIQP dataset as input (check the exact path in the Input panel).
2. Cells:

```python
!git clone https://github.com/<your-user>/mpqa-repro.git
%cd mpqa-repro
!pip install -q -r requirements.txt
```

```python
# Train (uses both GPUs automatically). Output lines flush immediately, so the log is readable while it runs.
!python main.py train --set data_dir=/kaggle/input/datasets/hakimdev/mriqp4/LabelMRI4 \
                         --set out_dir=/kaggle/working/run --set epochs=40
```

3. **Time limit.** A GPU commit ("Save & Run All") stops at about 12 hours and its output files may be lost if it does
   not finish. Measure one epoch first (about 6 minutes per epoch on one T4 at batch size 8) and choose `epochs` to
   finish in about 9-10 hours.
4. **Splitting a long run across commits.** `last.pth` (full state) is written after every epoch. Finish the first commit
   cleanly (for example `epochs=20`), add that version's output as an input of the next notebook, then:

```python
!python main.py train --set data_dir=... --set out_dir=/kaggle/working/run --set epochs=40 \
       --set resume_from=/kaggle/input/<previous-output>/run/last.pth
```

   `epochs` is the **total** target. The earlier `best.pth` next to `last.pth` is carried along automatically.
5. Smoke test of the head variants (about 25 minutes on 2 GPUs): `!python main.py smoke`
6. Evaluate a checkpoint: `!python main.py evaluate --ckpt /kaggle/working/run/best.pth --set data_dir=...`
7. The output of a finished commit appears on the version page (Notebook / Logs / Output tabs).
