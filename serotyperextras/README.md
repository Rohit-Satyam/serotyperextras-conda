## Serotyperextras

This is a conda package that tries to implement the Annotated Genome Aligner (AGA) for command line usage. The output alignment then can be used to annotate the new viral genomes.

```
python fastaga.py --reference /data/ksa_dengue/db/denv_refs/denv1.gb --query Sample105.fasta --mode global --out temp.txt --aa-matrix BLOSUM62 --protein-out aga_alignment_proteins.faa
```

Then annotate the query genome using
```
python annotate.py --reference-genbank sequence.gb --query-fasta sequence.fasta --aga-alignment tt --out-prefix temp --translatable-features CDS,mat_peptide
```
 
