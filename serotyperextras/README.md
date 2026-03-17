## Serotyperextras

This is a conda package that tries to implement the Annotated Genome Aligner (AGA) for command line usage. The output alignment then can be used to annotate the new viral genomes.

```
python fastaga.py --reference sequence.gb --query sequence.fasta --mode global --out alignment.txt --aa-matrix BLOSUM50
```

Then annotate the query genome using
```
python annotate.py --reference-genbank sequence.gb --query-fasta sequence.fasta --aga-alignment tt --out-prefix temp --translatable-features CDS,mat_peptide
```
 
