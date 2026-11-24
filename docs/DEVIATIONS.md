# Where this code differs from the paper

Reference: Huang et al., *Multi-dimensional perceptual quality assessment for magnetic resonance images*,
Health Information Science and Systems 13:65 (2025). Where the paper is silent, "released code" means the authors'
repository (github.com/bymianbao/MPQA).

| Item | Paper | This repo | Why |
|---|---|---|---|
| Head activation / dropout | two MLPs, details not given | GELU, dropout 0 (config: `act`, `head_drop`) | The released code uses ReLU + dropout 0.5. That setup **collapsed the four per-dimension scores** in our run (see below). |
| Where the QP (ZLPR) loss is applied | Eq. 5: the predicted quality scores | the **weight logits** | Follows the released code. It is also what makes Fig. 4 work (weight about 1 on the affected dimensions). Applying ZLPR to the scores would push the scores of *degraded* dimensions up, against the ranking loss. |
| ZLPR form | Eq. 4, per sample | per sample, averaged over the batch (stable `logsumexp` form) | The released code sums over the whole batch inside the log, which makes the loss depend on batch size. |
| Overall score | weighted sum of the scores | `softmax(weights) · scores` | The paper does not mention the softmax; the released code uses it and Fig. 4 is consistent with it. |
| Epochs | 100 | 40 (config: `epochs`) | 100 epochs is about 10.6 h on a T4, too close to Kaggle's 12 h limit. Resume is built in. |
| Batch size | 32 | 8 pairs (= 16 images) | 32 pairs (64 images at 384 px) does not fit a T4. |
| Precision | not stated | fp16 autocast on CUDA, loss in fp32 | Speed. |
| Input normalisation | not stated | mean 0.5 / std 0.5 | What the pretrained timm ViT expects. The released code skips it. |
| Accuracy metric | "overall accuracy" | accuracy on pairs with an expert preference (label != 0), by the sign of the score difference | **Inferred**: the paper's PSNR value (67.01) equals correct / non-tied pairs on our test split. No evaluation code is released. |
| Dimension labels per degradation | not specified | noise to SNR; contrast to contrast; crop to resolution; motion and herringbone to artifacts; masking to resolution + artifacts; denoise to SNR + resolution | Copied from the released code; only used as a training target, never as a model input. |
| Train/test split | random 4:1 | the CSVs shipped with the Kaggle copy of MRIQP; train/val split by subject | One subject appears in both CSVs (no shared scan or image file). `drop_shared_subjects=true` removes it from the test set. |
| Label order | overall first in Eq. 3, indexed as the 5th element in Eq. 7 | overall last everywhere | No effect on results. |

## The collapse finding

With the released settings (ReLU, dropout 0.5), the per-dimension scores of the two images in a pair became (almost)
identical, so the dimension accuracies fell to 1-27% while the Overall score still reached 78%. Measured on the
trained model: for 84.4% of images **every** hidden unit of the score head was 0 (only 19 of 256 units were ever
active) and the score difference was exactly 0 for about 71.5% of pairs. Independent dropout noise on the two images
makes the "equal" pairs expensive for any varying score, and constant scores are the cheapest way out; ReLU lets the
units die for good.

A 2x2 smoke test (activation x dropout, 6 epochs, 25% of the data, validation only) reproduced the collapse for the
original setting and showed that **removing the dropout fixes it** (also with ReLU); GELU was steadier on top of that.
See `docs/RESULTS.md`. This is evidence from a single seed and a short schedule; the full-data ablation is on the roadmap.
