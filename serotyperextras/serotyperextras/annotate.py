#!/usr/bin/env python3

"""
Project reference GenBank annotations through an AGA alignment onto a query sequence.
Inputs:
  --reference-genbank   Reference annotated genome in GenBank format
  --query-fasta         Query nucleotide FASTA
  --aga-alignment       Alignment text file from the traceback-producing AGA script
  --out-prefix          Output prefix

Outputs:
  <prefix>.gff3         Projected annotation in GFF3
  <prefix>.gb           Query GenBank with projected features
  <prefix>.faa          Protein FASTA translated from projected CDS
  <prefix>.fna          Projected CDS nucleotide FASTA

Dependencies:
  pip install biopython

Notes:
- This performs annotation projection through an existing alignment.
- It supports simple and compound GenBank locations.
- It handles strand.
- For CDS translation, it uses the projected query sequence.
- If projected CDS length is not divisible by 3 or contains internal stops, that is reported in qualifiers/headers.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, FeatureLocation, CompoundLocation
from Bio.SeqRecord import SeqRecord
# ---------------------------------------------------------------------
# AGA alignment parsing
# ---------------------------------------------------------------------

def parse_aga_alignment(path: str) -> Tuple[str, str]:
    """
    Parse alignment blocks of the form:
    # score=...
    # ref_start=...
    # qry_start=...
    U88536.1        AGTTG...
                    |||| |
    PZ055465.1      ----AG...
    Returns
    -------
    aligned_ref : str
    aligned_query : str
    """
    ref_chunks = []
    qry_chunks = []

    with open(path) as fh:
        lines = [line.rstrip("\n") for line in fh]

    # Keep blank and marker lines; remove only comments
    lines = [line for line in lines if not line.startswith("#")]

    i = 0
    n = len(lines)

    while i < n:
        # skip blank separators
        while i < n and lines[i].strip() == "":
            i += 1
        if i >= n:
            break

        if i + 2 >= n:
            raise ValueError(f"Incomplete alignment block near line {i+1}")

        ref_line = lines[i]
        # mid_line = lines[i + 1]
        qry_line = lines[i + 2]

        # reference/query lines must contain name + sequence
        ref_parts = ref_line.split(None, 1)
        qry_parts = qry_line.split(None, 1)

        if len(ref_parts) != 2:
            raise ValueError(f"Could not parse reference line {i+1}: {ref_line!r}")
        if len(qry_parts) != 2:
            raise ValueError(f"Could not parse query line {i+3}: {qry_line!r}")

        ref_seq = ref_parts[1].replace(" ", "")
        qry_seq = qry_parts[1].replace(" ", "")

        if len(ref_seq) != len(qry_seq):
            raise ValueError(
                f"Alignment block length mismatch near line {i+1}: "
                f"{len(ref_seq)} vs {len(qry_seq)}\n"
                f"REF: {ref_line}\n"
                f"QRY: {qry_line}"
            )

        ref_chunks.append(ref_seq)
        qry_chunks.append(qry_seq)

        i += 3

        # skip trailing blank lines between blocks
        while i < n and lines[i].strip() == "":
            i += 1

    aligned_ref = "".join(ref_chunks)
    aligned_query = "".join(qry_chunks)

    if len(aligned_ref) != len(aligned_query):
        raise ValueError(
            f"Parsed alignment lengths do not match: "
            f"{len(aligned_ref)} vs {len(aligned_query)}"
        )

    return aligned_ref, aligned_query

# ---------------------------------------------------------------------
# Reference/query loading
# ---------------------------------------------------------------------

def load_single_genbank(path: str) -> SeqRecord:
    recs = list(SeqIO.parse(path, "genbank"))
    if len(recs) != 1:
        raise ValueError(f"Expected 1 GenBank record in {path}, found {len(recs)}")
    return recs[0]

def load_single_fasta(path: str) -> SeqRecord:
    recs = list(SeqIO.parse(path, "fasta"))
    if len(recs) != 1:
        raise ValueError(f"Expected 1 FASTA record in {path}, found {len(recs)}")
    return recs[0]

# ---------------------------------------------------------------------
# Alignment coordinate mapping
# ---------------------------------------------------------------------

@dataclass
class AlignmentMap:
    ref_to_query: Dict[int, Optional[int]]
    query_to_ref: Dict[int, Optional[int]]
    ref_aln_cols: Dict[int, int]
    query_aln_cols: Dict[int, int]

def build_alignment_map(aligned_ref: str, aligned_query: str) -> AlignmentMap:
    """
    Build mappings:
      reference 1-based position -> query 1-based position or None
      query 1-based position -> reference 1-based position or None
    """
    ref_to_query: Dict[int, Optional[int]] = {}
    query_to_ref: Dict[int, Optional[int]] = {}
    ref_aln_cols: Dict[int, int] = {}
    query_aln_cols: Dict[int, int] = {}
    rpos = 0
    qpos = 0

    for col, (r, q) in enumerate(zip(aligned_ref, aligned_query), start=1):
        if r != "-":
            rpos += 1
            ref_aln_cols[rpos] = col
        if q != "-":
            qpos += 1
            query_aln_cols[qpos] = col
        if r != "-":
            ref_to_query[rpos] = qpos if q != "-" else None
        if q != "-":
            query_to_ref[qpos] = rpos if r != "-" else None
    return AlignmentMap(
        ref_to_query=ref_to_query,
        query_to_ref=query_to_ref,
        ref_aln_cols=ref_aln_cols,
        query_aln_cols=query_aln_cols,
    )

# ---------------------------------------------------------------------
# Feature projection helpers
# ---------------------------------------------------------------------

def feature_parts_in_genomic_order(feature: SeqFeature) -> List[FeatureLocation]:
    loc = feature.location
    if isinstance(loc, CompoundLocation):
        parts = list(loc.parts)
    else:
        parts = [loc]
    # Genomic order, not transcription order
    parts = sorted(parts, key=lambda p: int(p.start))
    return parts

def positions_from_location(location: FeatureLocation) -> List[int]:
    """
    Return 1-based genomic positions covered by a FeatureLocation.
    GenBank/Biopython locations are 0-based, end-exclusive.
    """
    return list(range(int(location.start) + 1, int(location.end) + 1))

def compress_positions_to_intervals(pos: List[int]) -> List[Tuple[int, int]]:
    """
    Convert sorted 1-based positions into 1-based closed intervals.
    """
    if not pos:
        return []
    intervals = []
    s = pos[0]
    e = pos[0]
    for p in pos[1:]:
        if p == e + 1:
            e = p
        else:
            intervals.append((s, e))
            s = e = p
    intervals.append((s, e))
    return intervals

def project_positions(ref_positions: Sequence[int], amap: AlignmentMap) -> List[int]:
    """
    Project reference genomic positions to query positions, dropping deleted bases.
    """
    qpos = []
    for rp in ref_positions:
        qp = amap.ref_to_query.get(rp, None)
        if qp is not None:
            qpos.append(qp)
    return qpos

def parts_to_projected_query_intervals(feature: SeqFeature, amap: AlignmentMap) -> List[Tuple[int, int]]:
    """
    Project each feature part independently to query intervals.
    Output: 1-based closed query intervals in genomic order.
    """
    projected = []
    for part in feature_parts_in_genomic_order(feature):
        ref_pos = positions_from_location(part)
        qry_pos = project_positions(ref_pos, amap)
        qry_pos = sorted(set(qry_pos))
        projected.extend(compress_positions_to_intervals(qry_pos))
    return projected

def make_biopython_location(intervals_1based_closed: List[Tuple[int, int]], strand: int):
    """
    Convert 1-based closed intervals to Biopython FeatureLocation / CompoundLocation.
    """
    if not intervals_1based_closed:
        return None
    parts = []
    for s, e in intervals_1based_closed:
        parts.append(FeatureLocation(s - 1, e, strand=strand))

    if len(parts) == 1:
        return parts[0]

    return CompoundLocation(parts)

def extract_intervals_sequence(query_seq: str, intervals: List[Tuple[int, int]], strand: int) -> str:
    """
    Extract concatenated nucleotide sequence from query according to projected intervals.
    Intervals must be given in genomic order (ascending).
    For negative strand CDS, concatenate genomic-order pieces first then reverse-complement.
    """
    seq = "".join(query_seq[s - 1:e] for s, e in intervals)
    if strand == -1:
        seq = str(Seq(seq).reverse_complement())
    return seq

def get_codon_start(feature: SeqFeature) -> int:
    """
    GenBank codon_start qualifier is 1,2,3. Default 1.
    """
    vals = feature.qualifiers.get("codon_start", ["1"])

    try:
        x = int(vals[0])
        if x in (1, 2, 3):
            return x
    except Exception:
        pass
    return 1

def translate_projected_cds(nt_seq: str, codon_start: int = 1, transl_table: int = 1) -> Tuple[str, List[str]]:
    """
    Translate projected CDS sequence.
    Returns protein sequence and warning list.
    """
    warnings = []

    trimmed = nt_seq[codon_start - 1:]
    if len(trimmed) % 3 != 0:
        warnings.append("CDS length not divisible by 3 after codon_start adjustment")

    usable_len = (len(trimmed) // 3) * 3
    translatable = trimmed[:usable_len]

    if usable_len == 0:
        return "", warnings + ["No translatable codons"]

    prot = str(Seq(translatable).translate(table=transl_table, to_stop=False))

    if "*" in prot[:-1]:
        warnings.append("Internal stop codon detected")

    if not nt_seq:
        warnings.append("Projected CDS empty")

    return prot, warnings

def calc_gff_phase(feature: SeqFeature) -> int:
    """
    GFF3 phase:
      0 means first base of feature is first base of codon
      1 means one base remains from previous codon
      2 means two bases remain from previous codon
    Approximate from codon_start for CDS start feature.
    """
    codon_start = get_codon_start(feature)
    return (codon_start - 1) % 3

# ---------------------------------------------------------------------
# GFF3 writing
# ---------------------------------------------------------------------

def gff3_escape(value: str) -> str:
    return (
        value.replace("%", "%25")
        .replace(";", "%3B")
        .replace("=", "%3D")
        .replace("&", "%26")
        .replace(",", "%2C")
        .replace("\t", " ")
    )

def feature_id(feature: SeqFeature, fallback: str) -> str:
    for key in ("protein_id", "locus_tag", "gene", "product"):
        if key in feature.qualifiers:
            val = str(feature.qualifiers[key][0]).strip()
            if val:
                return val.replace(" ", "_")
    return fallback

def write_gff3(
    out_path: str,
    query_record: SeqRecord,
    projected_features: List[Tuple[SeqFeature, List[Tuple[int, int]], str, str, List[str]]],
) -> None:
    """
    projected_features entries:
      (original_feature, projected_intervals, feature_id, protein_seq, warnings)
    """
    with open(out_path, "w") as out:
        out.write("##gff-version 3\n")
        out.write(f"##sequence-region {query_record.id} 1 {len(query_record.seq)}\n")

        for idx, (feat, intervals, fid, protein_seq, warnings) in enumerate(projected_features, start=1):
            if not intervals:
                continue

            strand = feat.location.strand or 1
            strand_char = "+" if strand == 1 else "-"
            source = "AGA_projection"
            seqid = query_record.id

            gene_name = feat.qualifiers.get("gene", [fid])[0]
            product = feat.qualifiers.get("product", ["projected_CDS"])[0]
            phase0 = calc_gff_phase(feat)
            parent_id = f"{fid}.{feat.type}"

            for part_i, (s, e) in enumerate(intervals, start=1):
                attrs = {
                    "ID": f"{parent_id}.part{part_i}",
                    "Parent": parent_id,
                    "gene": gene_name,
                    "product": product,
                }

                if warnings:
                    attrs["Note"] = "|".join(warnings)

                attr_txt = ";".join(f"{k}={gff3_escape(str(v))}" for k, v in attrs.items())
                phase = phase0 if part_i == 1 else 0
                out.write(
                    f"{seqid}\t{source}\t{feat.type}\t{s}\t{e}\t.\t{strand_char}\t{phase}\t{attr_txt}\n"
                )

# ---------------------------------------------------------------------
# GenBank writing
# ---------------------------------------------------------------------

def projected_feature_copy(
    original_feature: SeqFeature,
    projected_intervals: List[Tuple[int, int]],
    protein_seq: str,
    warnings: List[str],
) -> Optional[SeqFeature]:
    if not projected_intervals:
        return None

    strand = original_feature.location.strand or 1
    new_loc = make_biopython_location(projected_intervals, strand=strand)
    if new_loc is None:
        return None

    quals = {k: list(v) for k, v in original_feature.qualifiers.items()}

    if protein_seq:
        quals["translation"] = [protein_seq]

    if warnings:
        old_note = quals.get("note", [])
        quals["note"] = old_note + [f"AGA_projection_warning: {w}" for w in warnings]

    return SeqFeature(location=new_loc, type=original_feature.type, qualifiers=quals)

def write_genbank(out_path: str, query_record: SeqRecord, features: List[SeqFeature]) -> None:
    out_rec = SeqRecord(
        seq=query_record.seq,
        id=query_record.id,
        name=query_record.name,
        description=f"{query_record.description} projected annotation via AGA alignment",
    )
    out_rec.annotations["molecule_type"] = "DNA"
    out_rec.features = features
    SeqIO.write(out_rec, out_path, "genbank")

# ---------------------------------------------------------------------
# FASTA outputs
# ---------------------------------------------------------------------

def wrap_fasta(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i:i+width] for i in range(0, len(seq), width))

def write_fasta_records(records: List[Tuple[str, str]], out_path: str) -> None:
    with open(out_path, "w") as out:
        for header, seq in records:
            out.write(f">{header}\n")
            out.write(wrap_fasta(seq) + "\n")

# ---------------------------------------------------------------------
# Main projection logic
# ---------------------------------------------------------------------


def project_protein_coding_features(
    ref_record: SeqRecord,
    query_record: SeqRecord,
    amap: AlignmentMap,
    translatable_features,
) -> Tuple[List[Tuple[SeqFeature, List[Tuple[int, int]], str, str, List[str]]], List[SeqFeature], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Returns:
      projected_info
      genbank_features
      protein_fasta_records
      cds_fasta_records
    """

    projected_info = []
    genbank_features = []
    protein_fasta_records = []
    cds_fasta_records = []

    for idx, feat in enumerate(ref_record.features, start=1):
        if feat.type not in translatable_features:
            continue

        fid = feature_id(feat, f"CDS_{idx}")
        product = feat.qualifiers.get("product", ["projected_CDS"])[0]
        strand = feat.location.strand or 1

        projected_intervals = parts_to_projected_query_intervals(feat, amap)
        if not projected_intervals:
            continue

        nt_seq = extract_intervals_sequence(str(query_record.seq), projected_intervals, strand=strand)
        transl_table = 1
        if "transl_table" in feat.qualifiers:
            try:
                transl_table = int(feat.qualifiers["transl_table"][0])
            except Exception:
                transl_table = 1

        codon_start = get_codon_start(feat)
        protein_seq, warnings = translate_projected_cds(
            nt_seq,
            codon_start=codon_start,
            transl_table=transl_table,
        )


        # Additional alignment-aware warning

        ref_len = 0
        for p in feature_parts_in_genomic_order(feat):
            ref_len += len(positions_from_location(p))

        proj_len = len(nt_seq)
        if proj_len != ref_len:
            warnings.append(
                f"Projected feature length differs from reference feature ({proj_len} vs {ref_len})"
            )

        projected_info.append((feat, projected_intervals, fid, protein_seq, warnings))

        gb_feat = projected_feature_copy(feat, projected_intervals, protein_seq, warnings)

        if gb_feat is not None:

            genbank_features.append(gb_feat)

        protein_header = f"{fid} product={product}"

        if warnings:
            protein_header += " warnings=" + "|".join(warnings)

        protein_fasta_records.append((protein_header, protein_seq if protein_seq else ""))

        cds_header = f"{fid} product={product}"

        if warnings:
            cds_header += " warnings=" + "|".join(warnings)
        cds_fasta_records.append((cds_header, nt_seq))

    return projected_info, genbank_features, protein_fasta_records, cds_fasta_records

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Project reference GenBank annotations through an AGA alignment")
    p.add_argument("--reference-genbank", required=True, help="Reference annotated genome in GenBank format")
    p.add_argument("--query-fasta", required=True, help="Query nucleotide FASTA")
    p.add_argument("--aga-alignment", required=True, help="AGA alignment text output from traceback-capable script")
    p.add_argument("--out-prefix", required=True, help="Output prefix") 
    p.add_argument("--translatable-features", default="CDS,mat_peptide", help="Comma-separated feature types to translate/project, e.g. CDS,mat_peptide")
    return p.parse_args()

def main():
    args = parse_args()
    translatable_features = {
        x.strip() for x in args.translatable_features.split(",") if x.strip()
    }
    ref_record = load_single_genbank(args.reference_genbank)
    query_record = load_single_fasta(args.query_fasta)
    aligned_ref, aligned_query = parse_aga_alignment(args.aga_alignment)

    # Basic sanity
    ref_ungapped = aligned_ref.replace("-", "").upper()
    qry_ungapped = aligned_query.replace("-", "").upper()

    if ref_ungapped != str(ref_record.seq).upper():
        raise ValueError(
            "Ungapped aligned reference does not match the reference GenBank sequence.\n"
            "Make sure the alignment was produced against this exact reference."
        )

    if qry_ungapped != str(query_record.seq).upper():
        raise ValueError(
            "Ungapped aligned query does not match the query FASTA sequence.\n"
            "Make sure the alignment was produced against this exact query."
        )

    amap = build_alignment_map(aligned_ref, aligned_query)
    projected_info, gb_features, protein_records, cds_records = project_protein_coding_features(
        ref_record, query_record, amap, translatable_features
    )

    prefix = args.out_prefix
    gff3_path = prefix + ".gff3"
    gb_path = prefix + ".gb"
    faa_path = prefix + ".faa"
    fna_path = prefix + ".fna"

    write_gff3(gff3_path, query_record, projected_info)
    write_genbank(gb_path, query_record, gb_features)
    write_fasta_records(protein_records, faa_path)
    write_fasta_records(cds_records, fna_path)

    print(f"Wrote: {gff3_path}")
    print(f"Wrote: {gb_path}")
    print(f"Wrote: {faa_path}")
    print(f"Wrote: {fna_path}")


if __name__ == "__main__":
    main()
