# Notice on Third-Party Baseline Methods

This repository is prepared for public GitHub release of the proposed TiDAL-Net method. The comparison methods used in the manuscript are important for reproducibility, but their official source code may be subject to copyright, author rights, conference/journal supplementary-material rules, or repository-specific licenses.

Therefore, the following methods are **not redistributed** in this repository:

1. Autoformer
2. PAREformer
3. GNN-DTAN
4. MSD-GNN
5. MAG
6. HiSTAR
7. MSTGAD
8. TraverseNet
9. COOL
10. DSTGNN

For each baseline, an independent adapter file is provided under `baselines/`. These files are intentionally minimal and do not contain copied third-party implementations. To reproduce the comparison experiments, users should:

1. Carefully read the original paper of each baseline.
2. Obtain the official implementation from the original authors or official repository.
3. Follow the corresponding license and citation requirements.
4. Implement a local wrapper in the corresponding adapter file.
5. Keep the input/output interface consistent with TiDAL-Net for fair comparison.

This policy prevents unauthorized redistribution while preserving a clear experimental interface for reproducible research.
