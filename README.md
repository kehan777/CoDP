# CoDP (Contrastive-learning-based Distogram Prediction Model)
<img width="2126" height="324" alt="supple_new_画板 1-01" src="./fig/CoDP.png" />

Integrated with HalluDesign, the CoDP model serves as a fast, distogram-based ranking tool. It leverages contrastive learning to quickly screen sequences by their predicted structural compatibility, facilitating a more efficient design cycle. This capability directly enhances the stability of monomeric proteins and improves the specificity of protein-ligand interactions.

we also implement modified LigandMPNN in our code.

# Installation
if your already installied HalluDesign, you no need to install this.
```
mamba create -n CoDP python==3.11
conda activate CoDP
# for LigandmpnnY
cd LigandMPNN 
pip install -r requirements.txt
pip install transformers==4.49.0
bash get_model_params.sh "./model_params"
```
# Inference
```
python CoDP_MPNN.py --input_file .pdb --output_dir ./test --mpnn ligand_mpnn --esmhead
# you need to download 
CoDP_MPNN.py [-h] [--pdb_list PDB_LIST] [--input_file INPUT_FILE] [--fix_res_index FIX_RES_INDEX]
                    [--fix_chain_index FIX_CHAIN_INDEX] --output_dir OUTPUT_DIR [--num_seqs NUM_SEQS] [--esmhead]
                    [--mpnn MPNN] [--mpnn_temperature MPNN_TEMPERATURE]

```
# Training
the dataset we used for training are not able to realase now. But we release our training code.
```
python train_contact.py
python train_contractive.py
```
# Reference
```
@article {Fang2025.11.08.686881,
	author = {Fang, Minchao and Wang, Chentong and Shi, Jungang and Lian, Fangbai and Jin, Qihan and Wang, Zhe and Zhang, Yanzhe and Cui, Zhanyuan and Wang, YanJun and Ke, Yitao and Han, Qingzheng and Cao, Longxing},
	title = {HalluDesign: Protein Optimization and de novo Design via Iterative Structure Hallucination and Sequence design},
	elocation-id = {2025.11.08.686881},
	year = {2025},
	doi = {10.1101/2025.11.08.686881},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Deep learning has revolutionized biomolecular modeling, enabling the prediction of diverse structures with atomic accuracy. However, leveraging the atomic-level precision of the structure prediction model for de novo design remains challenging. Here, we present HalluDesign, a general all-atom framework for protein optimization and de novo design, which iteratively update protein structure and sequence. HalluDesign harnesses the inherent hallucination capabilities of AlphaFold3-style structure prediction models and enables fine-tune free, forward-pass only sequence-structure co-optimization. Structure conditioning at different noise level in the structure prediction stage allows precise control over the sampling space, facilitating tasks from local and global protein optimization to de novo design. We demonstrate the versatility of this approach by optimizing suboptimal structures, rescuing previously unsuccessful designs, designing new biomolecular interactions and generating new protein structures from scratch. Experimental characterization of binder design spanning small molecule, metal ion, protein, and antibody design of phosphorylation-specific peptide revealed high design success rates and excellent structural accuracy. Together, our comprehensive computational and experimental results highlight the broad utility of this framework. We anticipate that HalluDesign will further unlock the modeling and design potential of AlphaFold3-like models, enabling the systematic creation of complex proteins for a wide range of biotechnological applications.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2025/11/09/2025.11.08.686881},
	eprint = {https://www.biorxiv.org/content/early/2025/11/09/2025.11.08.686881.full.pdf},
	journal = {bioRxiv}
}
@article{dauparas2023atomic,
  title={Atomic context-conditioned protein sequence design using LigandMPNN},
  author={Dauparas, Justas and Lee, Gyu Rie and Pecoraro, Robert and An, Linna and Anishchenko, Ivan and Glasscock, Cameron and Baker, David},
  journal={Biorxiv},
  pages={2023--12},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},  
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```
# License
The CoDP project, including both the source code and model weights, is licensed under the [MIT License](LICENSE)


LigandMPNN project (https://github.com/dauparas/LigandMPNN), is licensed under the [MIT License](LICENSE)