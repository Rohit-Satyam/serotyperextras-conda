#!/usr/bin/env python3

"""
Fast AGA-style annotated genome aligner with traceback
-----------------------------------------------------
Paper-based reimplementation of the AGA dynamic programming recurrence,
optimized compared with the original pure-Python version.

Features
- Reads annotated reference genome from GenBank
- Reads single-record query FASTA
- CDS-aware nucleotide + amino-acid scoring
- Global or local alignment
- Full traceback
- Output alignment to --out
- Post-traceback cleanup to collapse compensating opposite indels
  (e.g. ref: C---TGAT / qry: CTGA---T -> CTGAT / CTGAT)

Install:
    pip install biopython numpy numba

Example:
    python fastaga.py \
        --reference ref.gb \
        --query qry.fa \
        --out aga_alignment.txt \
        --mode global
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
from numba import njit
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import substitution_matrices
from Bio.SeqFeature import CompoundLocation, FeatureLocation

NEG_INF = -1e15
MAX_OCC_DEFAULT = 16
EPS = 1e-9

# State codes
ST_STOP = 0
ST_M = 1
ST_P1 = 2
ST_P2 = 3
ST_P3 = 4
ST_Q1 = 5
ST_Q2 = 6
ST_Q3 = 7

NT_TO_INT = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "U": 3,
    "N": 4,
}

AA_ALPHABET = list("ARNDCQEGHILKMFPSTWYVBZX*")
AA_TO_INT = {aa: i for i, aa in enumerate(AA_ALPHABET)}


def read_single_genbank(path: str):
    recs = list(SeqIO.parse(path, "genbank"))
    if len(recs) != 1:
        raise ValueError(f"Expected exactly one GenBank record in {path}, found {len(recs)}")
    return recs[0]


def read_single_fasta(path: str) -> Tuple[str, str]:
    recs = list(SeqIO.parse(path, "fasta"))
    if len(recs) != 1:
        raise ValueError(f"Expected exactly one FASTA record in {path}, found {len(recs)}")
    return recs[0].id, str(recs[0].seq)


def encode_nt_seq(seq: str) -> np.ndarray:
    arr = np.empty(len(seq), dtype=np.int8)
    for i, ch in enumerate(seq.upper()):
        arr[i] = NT_TO_INT.get(ch, 4)
    return arr


def aa_char_to_int(ch: str) -> int:
    return AA_TO_INT.get(ch, AA_TO_INT["X"])


def translate_codon_string(codon: str) -> int:
    try:
        aa = str(Seq(codon).translate(table=1, to_stop=False))
    except Exception:
        aa = "X"
    return aa_char_to_int(aa)


def build_nt_sub_matrix(match: float, mismatch: float) -> np.ndarray:
    mat = np.full((5, 5), mismatch, dtype=np.float64)
    for i in range(4):
        mat[i, i] = match
    return mat


def build_aa_sub_matrix(matrix_source: str = "BLOSUM62") -> np.ndarray:
    """
    Build amino-acid substitution matrix.

    matrix_source can be:
      - a Biopython built-in matrix name, e.g. BLOSUM62, PAM250
      - a path to a plain-text custom matrix file
    """
    try:
        if os.path.isfile(matrix_source):
            with open(matrix_source) as fh:
                raw = substitution_matrices.read(fh, dtype=float)
        else:
            raw = substitution_matrices.load(matrix_source)
    except FileNotFoundError:
        raise ValueError(
            f"Amino-acid matrix '{matrix_source}' was not found.\n"
            f"If this is meant to be a built-in Biopython matrix, it is not available in your installation.\n"
            f"Use an available built-in matrix such as BLOSUM62, PAM250, etc.,\n"
            f"or provide a path to a custom matrix file."
        )

    mat = np.full((len(AA_ALPHABET), len(AA_ALPHABET)), -1.0, dtype=np.float64)

    for a, i in AA_TO_INT.items():
        for b, j in AA_TO_INT.items():
            try:
                mat[i, j] = float(raw[a, b])
            except Exception:
                mat[i, j] = 1.0 if a == b else -1.0

    return mat


def precompute_query_codon_aas(query_seq: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    q_fwd_start[n]: AA for query codon starting at 1-based n
    q_rev_end[n]:   AA for reverse-strand codon ending at 1-based n
    Invalid => -1
    """
    N = len(query_seq)
    q = query_seq.upper()
    q_fwd_start = np.full(N + 1, -1, dtype=np.int16)
    q_rev_end = np.full(N + 1, -1, dtype=np.int16)

    for n in range(1, N - 1):
        codon = q[n - 1:n + 2]
        if len(codon) == 3:
            q_fwd_start[n] = translate_codon_string(codon)

    for n in range(3, N + 1):
        codon = q[n - 3:n]
        codon_rc = str(Seq(codon).reverse_complement())
        q_rev_end[n] = translate_codon_string(codon_rc)

    return q_fwd_start, q_rev_end


def _extract_parts(feature) -> List[FeatureLocation]:
    loc = feature.location
    if isinstance(loc, CompoundLocation):
        return list(loc.parts)
    return [loc]


def _positions_in_translation_order(feature) -> List[int]:
    parts = _extract_parts(feature)
    strand = feature.location.strand or 1

    if strand == 1:
        pos = []
        for p in parts:
            pos.extend(range(int(p.start) + 1, int(p.end) + 1))
        return pos

    pos = []
    for p in reversed(parts):
        pos.extend(range(int(p.end), int(p.start), -1))
    return pos

def _feature_label(feat, idx: int) -> str:
    gene = feat.qualifiers.get("gene", [""])[0]
    product = feat.qualifiers.get("product", [""])[0]
    protein_id = feat.qualifiers.get("protein_id", [""])[0]

    parts = [x for x in [gene, product, protein_id] if x]
    if parts:
        return " | ".join(parts)
    return f"CDS_{idx}"


def extract_cds_entries(record):
    """
    Extract CDS entries in translation order.
    Returns a list of dicts:
      {
        'id': 'CDS_1',
        'label': 'gene | product | protein_id',
        'positions': [1-based genomic positions in translation order],
        'strand': +1 / -1
      }
    """
    cds_entries = []
    idx = 0

    for feat in record.features:
        if feat.type != "CDS":
            continue

        idx += 1
        positions = _positions_in_translation_order(feat)
        positions = positions[: (len(positions) // 3) * 3]
        if not positions:
            continue

        cds_entries.append({
            "id": f"CDS_{idx}",
            "label": _feature_label(feat, idx),
            "positions": positions,
            "strand": feat.location.strand or 1,
        })

    return cds_entries


def build_alignment_ref_pos_map(aligned_ref: str):
    """
    For each alignment column, return the 1-based reference genomic position
    represented in that column, or None if ref has a gap there.
    """
    ref_map = []
    ref_pos = 0
    for ch in aligned_ref:
        if ch == "-":
            ref_map.append(None)
        else:
            ref_pos += 1
            ref_map.append(ref_pos)
    return ref_map


def translate_aligned_codon(codon: str) -> str:
    """
    Translate a codon extracted from an aligned nucleotide sequence.

    Rules:
    - '---' -> '-'
    - any codon with gaps or N or wrong length -> 'X'
    - otherwise standard translation
    """
    codon = codon.upper()
    if codon == "---":
        return "-"
    if len(codon) != 3:
        return "X"
    if "-" in codon or "N" in codon:
        return "X"
    try:
        return str(Seq(codon).translate(table=1, to_stop=False))
    except Exception:
        return "X"


def build_cds_protein_alignment(aligned_ref: str, aligned_qry: str, ref_pos_map, cds_entry):
    """
    Build protein alignment for one CDS from the final nucleotide alignment.

    We use reference genomic positions to select the alignment columns that belong
    to this CDS, preserving translation order for both strands.
    """
    pos_to_col = {}
    for col, gpos in enumerate(ref_pos_map):
        if gpos is not None:
            pos_to_col[gpos] = col

    ref_nt = []
    qry_nt = []

    for gpos in cds_entry["positions"]:
        if gpos not in pos_to_col:
            continue
        col = pos_to_col[gpos]
        ref_nt.append(aligned_ref[col])
        qry_nt.append(aligned_qry[col])

    usable = (len(ref_nt) // 3) * 3
    ref_nt = ref_nt[:usable]
    qry_nt = qry_nt[:usable]

    ref_aa = []
    qry_aa = []

    for i in range(0, usable, 3):
        ref_codon = "".join(ref_nt[i:i+3])
        qry_codon = "".join(qry_nt[i:i+3])

        ref_aa.append(translate_aligned_codon(ref_codon))
        qry_aa.append(translate_aligned_codon(qry_codon))

    return "".join(ref_aa), "".join(qry_aa)


def write_protein_alignments(
    out_path: str,
    record,
    qry_name: str,
    aligned_ref: str,
    aligned_qry: str,
    width: int = 100
) -> None:
    """
    Write one protein alignment per CDS, derived from the final nucleotide alignment.
    Output format is simple FASTA with paired ref/qry records per CDS.
    """
    cds_entries = extract_cds_entries(record)
    ref_pos_map = build_alignment_ref_pos_map(aligned_ref)

    with open(out_path, "w") as out:
        for cds in cds_entries:
            ref_aa, qry_aa = build_cds_protein_alignment(
                aligned_ref, aligned_qry, ref_pos_map, cds
            )

            out.write(f">{record.id}|ref|{cds['id']}|{cds['label']}\n")
            for i in range(0, len(ref_aa), width):
                out.write(ref_aa[i:i+width] + "\n")

            out.write(f">{qry_name}|qry|{cds['id']}|{cds['label']}\n")
            for i in range(0, len(qry_aa), width):
                out.write(qry_aa[i:i+width] + "\n")
                
def build_annotation_arrays(record, max_occ: int = MAX_OCC_DEFAULT):
    """
    Dense per-position annotation arrays.
    For each genome position m, store up to max_occ codon participations:
      occ_count[m]
      occ_cpos[m, k]   codon position 1/2/3
      occ_strand[m, k] +1 / -1
      occ_aa[m, k]     translated amino acid of genomic codon
    """
    genome_seq = str(record.seq).upper()
    M = len(genome_seq)
    occ_count = np.zeros(M + 1, dtype=np.int16)
    occ_cpos = np.zeros((M + 1, max_occ), dtype=np.int8)
    occ_strand = np.zeros((M + 1, max_occ), dtype=np.int8)
    occ_aa = np.full((M + 1, max_occ), -1, dtype=np.int16)

    for feat in record.features:
        if feat.type != "CDS":
            continue

        strand = feat.location.strand or 1
        positions = _positions_in_translation_order(feat)
        positions = positions[: (len(positions) // 3) * 3]

        for i in range(0, len(positions), 3):
            codon_positions = positions[i:i + 3]
            if len(codon_positions) != 3:
                continue

            contiguous = (
                strand == 1
                and codon_positions[1] == codon_positions[0] + 1
                and codon_positions[2] == codon_positions[1] + 1
            ) or (
                strand == -1
                and codon_positions[1] == codon_positions[0] - 1
                and codon_positions[2] == codon_positions[1] - 1
            )

            if not contiguous:
                continue

            codon = "".join(genome_seq[p - 1] for p in codon_positions)
            if strand == -1:
                codon = str(Seq(codon).reverse_complement())
            aa_idx = translate_codon_string(codon)

            for cpos, gpos in enumerate(codon_positions, start=1):
                idx = occ_count[gpos]
                if idx >= max_occ:
                    raise ValueError(
                        f"Position {gpos} has more than {max_occ} CDS overlaps; increase --max-occ"
                    )
                occ_cpos[gpos, idx] = cpos
                occ_strand[gpos, idx] = strand
                occ_aa[gpos, idx] = aa_idx
                occ_count[gpos] += 1

    return occ_count, occ_cpos, occ_strand, occ_aa


@njit(cache=True)
def aa_match_for_occ(cpos, strand, n_match, N, q_fwd_start, q_rev_end):
    if strand == 1:
        start = n_match - cpos + 1
        if start < 1 or start > N - 2:
            return -1
        return q_fwd_start[start]
    else:
        endpos = n_match + cpos - 1
        if endpos < 3 or endpos > N:
            return -1
        return q_rev_end[endpos]


@njit(cache=True)
def chi_sum_for_pos(
    m, n, occ_count, occ_cpos, occ_strand, occ_aa,
    q_fwd_start, q_rev_end, aa_sub
):
    s = 0.0
    cnt = occ_count[m]
    N = len(q_fwd_start) - 1
    for k in range(cnt):
        cpos = occ_cpos[m, k]
        if cpos != 1:
            continue
        strand = occ_strand[m, k]
        aa_g = occ_aa[m, k]
        aa_q = aa_match_for_occ(cpos, strand, n, N, q_fwd_start, q_rev_end)
        if aa_q >= 0:
            s += aa_sub[aa_g, aa_q]
    return s


@njit(cache=True)
def omega_k(k, aa_gap_open, aa_gap_extend):
    if k % 3 == 1:
        aa_gap_len = (k - 1) // 3 + 1
        if aa_gap_len == 1:
            return aa_gap_open
        return aa_gap_extend
    return 0.0


@njit(cache=True)
def phi_k(k, frameshift_penalty):
    r = k % 3
    if r == 1:
        return frameshift_penalty
    elif r == 2:
        return 0.0
    else:
        return -frameshift_penalty


@njit(cache=True)
def eta_sum_for_pos(
    gpos, n_match, k_gap,
    occ_count, occ_cpos, occ_strand, occ_aa,
    q_fwd_start, q_rev_end, aa_sub,
    aa_gap_open, aa_gap_extend,
    frameshift_penalty, misaligned_codon_penalty
):
    cnt = occ_count[gpos]
    s = 0.0
    om = omega_k(k_gap, aa_gap_open, aa_gap_extend)
    ph = phi_k(k_gap, frameshift_penalty)
    N = len(q_fwd_start) - 1

    for i in range(cnt):
        cpos = occ_cpos[gpos, i]
        strand = occ_strand[gpos, i]
        aa_g = occ_aa[gpos, i]
        nu = 0.0

        if cpos != 1 and k_gap == 1:
            aa_q = aa_match_for_occ(cpos, strand, n_match, N, q_fwd_start, q_rev_end)
            prev_aa = 0.0
            if aa_q >= 0:
                prev_aa = aa_sub[aa_g, aa_q]
            nu = -prev_aa + misaligned_codon_penalty

        s += nu + om + ph

    return s


@njit(cache=True)
def state_priority(state):
    if state == ST_M:
        return 7
    if state == ST_P3 or state == ST_Q3:
        return 6
    if state == ST_P2 or state == ST_Q2:
        return 5
    if state == ST_P1 or state == ST_Q1:
        return 4
    return 0


@njit(cache=True)
def better_candidate(score, state, best_score, best_state):
    if score > best_score + EPS:
        return True
    if abs(score - best_score) <= EPS and state_priority(state) > state_priority(best_state):
        return True
    return False


@njit(cache=True)
def aga_fill_matrices(
    G, B,
    nt_sub, aa_sub,
    occ_count, occ_cpos, occ_strand, occ_aa,
    q_fwd_start, q_rev_end,
    nt_gap_open, nt_gap_extend,
    aa_gap_open, aa_gap_extend,
    aa_weight,
    frameshift_penalty,
    misaligned_codon_penalty,
    local_mode
):
    Mlen = len(G)
    Nlen = len(B)

    D = np.full((Mlen + 1, Nlen + 1), NEG_INF, dtype=np.float64)
    Ms = np.full((Mlen + 1, Nlen + 1), NEG_INF, dtype=np.float64)
    P1 = np.full((Mlen + 1, Nlen + 1), NEG_INF, dtype=np.float64)
    P2 = np.full((Mlen + 1, Nlen + 1), NEG_INF, dtype=np.float64)
    P3 = np.full((Mlen + 1, Nlen + 1), NEG_INF, dtype=np.float64)
    Q1 = np.full((Mlen + 1, Nlen + 1), NEG_INF, dtype=np.float64)
    Q2 = np.full((Mlen + 1, Nlen + 1), NEG_INF, dtype=np.float64)
    Q3 = np.full((Mlen + 1, Nlen + 1), NEG_INF, dtype=np.float64)

    trD = np.zeros((Mlen + 1, Nlen + 1), dtype=np.uint8)
    trP1 = np.zeros((Mlen + 1, Nlen + 1), dtype=np.uint8)
    trQ1 = np.zeros((Mlen + 1, Nlen + 1), dtype=np.uint8)

    best_score = 0.0 if local_mode else NEG_INF
    best_i = 0
    best_j = 0

    if local_mode:
        D[:, :] = 0.0
        Ms[:, :] = 0.0
        P1[:, :] = 0.0
        P2[:, :] = 0.0
        P3[:, :] = 0.0
        Q1[:, :] = 0.0
        Q2[:, :] = 0.0
        Q3[:, :] = 0.0
    else:
        D[0, 0] = 0.0

        for n in range(1, Nlen + 1):
            gpos = 1 if Mlen >= 1 else 0
            if n == 1:
                dp = nt_gap_open + aa_weight * eta_sum_for_pos(
                    gpos, 0, 1,
                    occ_count, occ_cpos, occ_strand, occ_aa,
                    q_fwd_start, q_rev_end, aa_sub,
                    aa_gap_open, aa_gap_extend,
                    frameshift_penalty, misaligned_codon_penalty
                )
                P1[0, n] = D[0, n - 1] + dp
                trP1[0, n] = ST_STOP
            elif n % 3 == 2:
                dp = nt_gap_extend + aa_weight * eta_sum_for_pos(
                    gpos, n - 1, 2,
                    occ_count, occ_cpos, occ_strand, occ_aa,
                    q_fwd_start, q_rev_end, aa_sub,
                    aa_gap_open, aa_gap_extend,
                    frameshift_penalty, misaligned_codon_penalty
                )
                P2[0, n] = P1[0, n - 1] + dp
            elif n % 3 == 0:
                dp = nt_gap_extend + aa_weight * eta_sum_for_pos(
                    gpos, n - 1, 3,
                    occ_count, occ_cpos, occ_strand, occ_aa,
                    q_fwd_start, q_rev_end, aa_sub,
                    aa_gap_open, aa_gap_extend,
                    frameshift_penalty, misaligned_codon_penalty
                )
                P3[0, n] = P2[0, n - 1] + dp
            else:
                dp_open = nt_gap_open + aa_weight * eta_sum_for_pos(
                    gpos, n - 1, 1,
                    occ_count, occ_cpos, occ_strand, occ_aa,
                    q_fwd_start, q_rev_end, aa_sub,
                    aa_gap_open, aa_gap_extend,
                    frameshift_penalty, misaligned_codon_penalty
                )
                dp_ext = nt_gap_extend + aa_weight * eta_sum_for_pos(
                    gpos, n - 1, 4,
                    occ_count, occ_cpos, occ_strand, occ_aa,
                    q_fwd_start, q_rev_end, aa_sub,
                    aa_gap_open, aa_gap_extend,
                    frameshift_penalty, misaligned_codon_penalty
                )
                fromD = D[0, n - 1] + dp_open
                fromP3 = P3[0, n - 1] + dp_ext
                if better_candidate(fromD, ST_STOP, fromP3, ST_P3):
                    P1[0, n] = fromD
                    trP1[0, n] = ST_STOP
                else:
                    P1[0, n] = fromP3
                    trP1[0, n] = ST_P3

            best = P1[0, n]
            src = ST_P1
            if better_candidate(P2[0, n], ST_P2, best, src):
                best = P2[0, n]
                src = ST_P2
            if better_candidate(P3[0, n], ST_P3, best, src):
                best = P3[0, n]
                src = ST_P3
            D[0, n] = best
            trD[0, n] = src

        for m in range(1, Mlen + 1):
            npos = 1 if Nlen >= 1 else 0
            if m == 1:
                dq = nt_gap_open + aa_weight * eta_sum_for_pos(
                    m, npos, 1,
                    occ_count, occ_cpos, occ_strand, occ_aa,
                    q_fwd_start, q_rev_end, aa_sub,
                    aa_gap_open, aa_gap_extend,
                    frameshift_penalty, misaligned_codon_penalty
                )
                Q1[m, 0] = D[m - 1, 0] + dq
                trQ1[m, 0] = ST_STOP
            elif m % 3 == 2:
                dq = nt_gap_extend + aa_weight * eta_sum_for_pos(
                    m, npos, 2,
                    occ_count, occ_cpos, occ_strand, occ_aa,
                    q_fwd_start, q_rev_end, aa_sub,
                    aa_gap_open, aa_gap_extend,
                    frameshift_penalty, misaligned_codon_penalty
                )
                Q2[m, 0] = Q1[m - 1, 0] + dq
            elif m % 3 == 0:
                dq = nt_gap_extend + aa_weight * eta_sum_for_pos(
                    m, npos, 3,
                    occ_count, occ_cpos, occ_strand, occ_aa,
                    q_fwd_start, q_rev_end, aa_sub,
                    aa_gap_open, aa_gap_extend,
                    frameshift_penalty, misaligned_codon_penalty
                )
                Q3[m, 0] = Q2[m - 1, 0] + dq
            else:
                dq_open = nt_gap_open + aa_weight * eta_sum_for_pos(
                    m, npos, 1,
                    occ_count, occ_cpos, occ_strand, occ_aa,
                    q_fwd_start, q_rev_end, aa_sub,
                    aa_gap_open, aa_gap_extend,
                    frameshift_penalty, misaligned_codon_penalty
                )
                dq_ext = nt_gap_extend + aa_weight * eta_sum_for_pos(
                    m, npos, 4,
                    occ_count, occ_cpos, occ_strand, occ_aa,
                    q_fwd_start, q_rev_end, aa_sub,
                    aa_gap_open, aa_gap_extend,
                    frameshift_penalty, misaligned_codon_penalty
                )
                fromD = D[m - 1, 0] + dq_open
                fromQ3 = Q3[m - 1, 0] + dq_ext
                if better_candidate(fromD, ST_STOP, fromQ3, ST_Q3):
                    Q1[m, 0] = fromD
                    trQ1[m, 0] = ST_STOP
                else:
                    Q1[m, 0] = fromQ3
                    trQ1[m, 0] = ST_Q3

            best = Q1[m, 0]
            src = ST_Q1
            if better_candidate(Q2[m, 0], ST_Q2, best, src):
                best = Q2[m, 0]
                src = ST_Q2
            if better_candidate(Q3[m, 0], ST_Q3, best, src):
                best = Q3[m, 0]
                src = ST_Q3
            D[m, 0] = best
            trD[m, 0] = src

    for m in range(1, Mlen + 1):
        for n in range(1, Nlen + 1):
            dmatch = nt_sub[G[m - 1], B[n - 1]]
            dmatch += aa_weight * chi_sum_for_pos(
                m, n, occ_count, occ_cpos, occ_strand, occ_aa,
                q_fwd_start, q_rev_end, aa_sub
            )

            Ms[m, n] = D[m - 1, n - 1] + dmatch

            gpos_for_p = m + 1 if m < Mlen else m
            npos_for_q = n + 1 if n < Nlen else n

            dp_open = nt_gap_open + aa_weight * eta_sum_for_pos(
                gpos_for_p, n, 1,
                occ_count, occ_cpos, occ_strand, occ_aa,
                q_fwd_start, q_rev_end, aa_sub,
                aa_gap_open, aa_gap_extend,
                frameshift_penalty, misaligned_codon_penalty
            )
            dp_ext = nt_gap_extend + aa_weight * eta_sum_for_pos(
                gpos_for_p, n, 4,
                occ_count, occ_cpos, occ_strand, occ_aa,
                q_fwd_start, q_rev_end, aa_sub,
                aa_gap_open, aa_gap_extend,
                frameshift_penalty, misaligned_codon_penalty
            )

            fromM = Ms[m, n - 1] + dp_open
            fromP3 = P3[m, n - 1] + dp_ext
            if better_candidate(fromM, ST_M, fromP3, ST_P3):
                P1[m, n] = fromM
                trP1[m, n] = ST_M
            else:
                P1[m, n] = fromP3
                trP1[m, n] = ST_P3

            dp2 = nt_gap_extend + aa_weight * eta_sum_for_pos(
                gpos_for_p, n, 2,
                occ_count, occ_cpos, occ_strand, occ_aa,
                q_fwd_start, q_rev_end, aa_sub,
                aa_gap_open, aa_gap_extend,
                frameshift_penalty, misaligned_codon_penalty
            )
            dp3 = nt_gap_extend + aa_weight * eta_sum_for_pos(
                gpos_for_p, n, 3,
                occ_count, occ_cpos, occ_strand, occ_aa,
                q_fwd_start, q_rev_end, aa_sub,
                aa_gap_open, aa_gap_extend,
                frameshift_penalty, misaligned_codon_penalty
            )

            P2[m, n] = P1[m, n - 1] + dp2
            P3[m, n] = P2[m, n - 1] + dp3

            dq_open = nt_gap_open + aa_weight * eta_sum_for_pos(
                m, npos_for_q, 1,
                occ_count, occ_cpos, occ_strand, occ_aa,
                q_fwd_start, q_rev_end, aa_sub,
                aa_gap_open, aa_gap_extend,
                frameshift_penalty, misaligned_codon_penalty
            )
            dq_ext = nt_gap_extend + aa_weight * eta_sum_for_pos(
                m, npos_for_q, 4,
                occ_count, occ_cpos, occ_strand, occ_aa,
                q_fwd_start, q_rev_end, aa_sub,
                aa_gap_open, aa_gap_extend,
                frameshift_penalty, misaligned_codon_penalty
            )

            fromM = Ms[m - 1, n] + dq_open
            fromQ3 = Q3[m - 1, n] + dq_ext
            if better_candidate(fromM, ST_M, fromQ3, ST_Q3):
                Q1[m, n] = fromM
                trQ1[m, n] = ST_M
            else:
                Q1[m, n] = fromQ3
                trQ1[m, n] = ST_Q3

            dq2 = nt_gap_extend + aa_weight * eta_sum_for_pos(
                m, npos_for_q, 2,
                occ_count, occ_cpos, occ_strand, occ_aa,
                q_fwd_start, q_rev_end, aa_sub,
                aa_gap_open, aa_gap_extend,
                frameshift_penalty, misaligned_codon_penalty
            )
            dq3 = nt_gap_extend + aa_weight * eta_sum_for_pos(
                m, npos_for_q, 3,
                occ_count, occ_cpos, occ_strand, occ_aa,
                q_fwd_start, q_rev_end, aa_sub,
                aa_gap_open, aa_gap_extend,
                frameshift_penalty, misaligned_codon_penalty
            )

            Q2[m, n] = Q1[m - 1, n] + dq2
            Q3[m, n] = Q2[m - 1, n] + dq3

            best = Ms[m, n]
            src = ST_M
            if better_candidate(P1[m, n], ST_P1, best, src):
                best = P1[m, n]
                src = ST_P1
            if better_candidate(P2[m, n], ST_P2, best, src):
                best = P2[m, n]
                src = ST_P2
            if better_candidate(P3[m, n], ST_P3, best, src):
                best = P3[m, n]
                src = ST_P3
            if better_candidate(Q1[m, n], ST_Q1, best, src):
                best = Q1[m, n]
                src = ST_Q1
            if better_candidate(Q2[m, n], ST_Q2, best, src):
                best = Q2[m, n]
                src = ST_Q2
            if better_candidate(Q3[m, n], ST_Q3, best, src):
                best = Q3[m, n]
                src = ST_Q3

            if local_mode and best < 0.0:
                D[m, n] = 0.0
                trD[m, n] = ST_STOP
            else:
                D[m, n] = best
                trD[m, n] = src

            if local_mode and D[m, n] > best_score:
                best_score = D[m, n]
                best_i = m
                best_j = n

    if not local_mode:
        best_score = D[Mlen, Nlen]
        best_i = Mlen
        best_j = Nlen

    return D, trD, trP1, trQ1, best_score, best_i, best_j


def traceback_alignment(
    G_str: str,
    B_str: str,
    trD: np.ndarray,
    trP1: np.ndarray,
    trQ1: np.ndarray,
    best_i: int,
    best_j: int,
    local_mode: bool
) -> Tuple[str, str, int, int]:
    i = best_i
    j = best_j
    aln_ref = []
    aln_qry = []

    state = int(trD[i, j])

    while i >= 0 and j >= 0:
        if state == ST_STOP:
            break

        if state == ST_M:
            if i <= 0 or j <= 0:
                break
            aln_ref.append(G_str[i - 1])
            aln_qry.append(B_str[j - 1])
            i -= 1
            j -= 1
            if i < 0 or j < 0:
                break
            state = int(trD[i, j])

        elif state == ST_P1:
            if j <= 0:
                break
            prev = int(trP1[i, j])
            aln_ref.append("-")
            aln_qry.append(B_str[j - 1])
            j -= 1
            state = prev

        elif state == ST_P2:
            if j <= 0:
                break
            aln_ref.append("-")
            aln_qry.append(B_str[j - 1])
            j -= 1
            state = ST_P1

        elif state == ST_P3:
            if j <= 0:
                break
            aln_ref.append("-")
            aln_qry.append(B_str[j - 1])
            j -= 1
            state = ST_P2

        elif state == ST_Q1:
            if i <= 0:
                break
            prev = int(trQ1[i, j])
            aln_ref.append(G_str[i - 1])
            aln_qry.append("-")
            i -= 1
            state = prev

        elif state == ST_Q2:
            if i <= 0:
                break
            aln_ref.append(G_str[i - 1])
            aln_qry.append("-")
            i -= 1
            state = ST_Q1

        elif state == ST_Q3:
            if i <= 0:
                break
            aln_ref.append(G_str[i - 1])
            aln_qry.append("-")
            i -= 1
            state = ST_Q2

        else:
            raise RuntimeError(f"Unknown traceback state: {state}")

        if local_mode and i >= 0 and j >= 0 and state == ST_STOP:
            break

    aln_ref.reverse()
    aln_qry.reverse()
    return "".join(aln_ref), "".join(aln_qry), i, j


def collapse_compensating_indels(aln_ref: str, aln_qry: str) -> Tuple[str, str]:
    """
    Collapse back-to-back opposite equal-length single-sided gap runs when
    the total ungapped sequence across the whole region is identical.

    Example:
      ref: C---TGAT
      qry: CTGA---T
    becomes:
      ref: CTGAT
      qry: CTGAT
    """
    r = aln_ref
    q = aln_qry

    changed = True
    while changed:
        changed = False
        i = 0
        n = len(r)

        while i < n:
            # first block: ref-gap only
            if r[i] == "-" and q[i] != "-":
                g1_start = i
                while i < n and r[i] == "-" and q[i] != "-":
                    i += 1
                g1_end = i
                g1_len = g1_end - g1_start

                mid_start = i
                while i < n and r[i] != "-" and q[i] != "-":
                    i += 1
                mid_end = i

                g2_start = i
                while i < n and q[i] == "-" and r[i] != "-":
                    i += 1
                g2_end = i
                g2_len = g2_end - g2_start

                if g1_len > 0 and g1_len == g2_len:
                    ref_piece = "".join(ch for ch in r[g1_start:g2_end] if ch != "-")
                    qry_piece = "".join(ch for ch in q[g1_start:g2_end] if ch != "-")
                    if ref_piece == qry_piece:
                        r = r[:g1_start] + ref_piece + r[g2_end:]
                        q = q[:g1_start] + qry_piece + q[g2_end:]
                        changed = True
                        break

            # first block: qry-gap only
            elif q[i] == "-" and r[i] != "-":
                g1_start = i
                while i < n and q[i] == "-" and r[i] != "-":
                    i += 1
                g1_end = i
                g1_len = g1_end - g1_start

                mid_start = i
                while i < n and r[i] != "-" and q[i] != "-":
                    i += 1
                mid_end = i

                g2_start = i
                while i < n and r[i] == "-" and q[i] != "-":
                    i += 1
                g2_end = i
                g2_len = g2_end - g2_start

                if g1_len > 0 and g1_len == g2_len:
                    ref_piece = "".join(ch for ch in r[g1_start:g2_end] if ch != "-")
                    qry_piece = "".join(ch for ch in q[g1_start:g2_end] if ch != "-")
                    if ref_piece == qry_piece:
                        r = r[:g1_start] + ref_piece + r[g2_end:]
                        q = q[:g1_start] + qry_piece + q[g2_end:]
                        changed = True
                        break
            else:
                i += 1

    return r, q


def cleanup_alignment(aln_ref: str, aln_qry: str, max_passes: int = 10) -> Tuple[str, str]:
    """
    Repeatedly collapse compensating opposite indels until stable.
    """
    r, q = aln_ref, aln_qry
    for _ in range(max_passes):
        new_r, new_q = collapse_compensating_indels(r, q)
        if new_r == r and new_q == q:
            break
        r, q = new_r, new_q
    return r, q


def alignment_stats(aln_ref: str, aln_qry: str) -> Tuple[int, int, int]:
    matches = 0
    mismatches = 0
    gap_cols = 0
    for a, b in zip(aln_ref, aln_qry):
        if a == "-" or b == "-":
            gap_cols += 1
        elif a == b:
            matches += 1
        else:
            mismatches += 1
    return matches, mismatches, gap_cols


def write_alignment(
    out_path: str,
    ref_name: str,
    qry_name: str,
    aligned_ref: str,
    aligned_qry: str,
    score: float,
    start_i: int,
    start_j: int,
    end_i: int,
    end_j: int,
    width: int = 100
) -> None:
    matches, mismatches, gap_cols = alignment_stats(aligned_ref, aligned_qry)

    with open(out_path, "w") as out:
        out.write(f"# score={score:.3f}\n")
        out.write(f"# ref_start={start_i} ref_end={end_i}\n")
        out.write(f"# qry_start={start_j} qry_end={end_j}\n")
        out.write(f"# aln_len={len(aligned_ref)} matches={matches} mismatches={mismatches} gap_cols={gap_cols}\n\n")

        for k in range(0, len(aligned_ref), width):
            r = aligned_ref[k:k + width]
            q = aligned_qry[k:k + width]
            mid = "".join("|" if a == b and a != "-" else " " for a, b in zip(r, q))
            out.write(f"{ref_name:<15} {r}\n")
            out.write(f"{'':<15} {mid}\n")
            out.write(f"{qry_name:<15} {q}\n\n")


def parse_args():
    p = argparse.ArgumentParser(description="Fast AGA-style annotated genome aligner with traceback")
    p.add_argument("--reference", required=True, help="Reference GenBank with CDS annotations")
    p.add_argument("--query", required=True, help="Single-record query FASTA")
    p.add_argument("--out", required=True, help="Output alignment text file")
    p.add_argument("--mode", choices=["global", "local"], default="global")
    p.add_argument("--protein-out", default=None, help="Optional FASTA output for CDS-wise protein alignments derived from the final nucleotide alignment")
    p.add_argument("--nt-match", type=float, default=2.0, help="default: 2")
    p.add_argument("--nt-mismatch", type=float, default=-2.0, help="default: -2")
    p.add_argument("--nt-gap-open", type=float, default=-10.0, help="default: -10")
    p.add_argument("--nt-gap-extend", type=float, default=-1.0, help="default: -1")

    p.add_argument(
        "--aa-matrix",
        default="BLOSUM62",
        help=(
            "Built-in matrix name or path to custom matrix file. Available built-ins include "
            "'BENNER22', 'BENNER6', 'BENNER74', 'BLASTN', 'BLASTP', 'BLOSUM45', 'BLOSUM50', "
            "'BLOSUM62', 'BLOSUM80', 'BLOSUM90', 'DAYHOFF', 'FENG', 'GENETIC', 'GONNET1992', "
            "'HOXD70', 'JOHNSON', 'JONES', 'LEVIN', 'MCLACHLAN', 'MDM78', 'MEGABLAST', "
            "'NUC.4.4', 'PAM250', 'PAM30', 'PAM70', 'RAO', 'RISLER', 'SCHNEIDER', 'STR', 'TRANS'"
        )
    )
    p.add_argument("--aa-gap-open", type=float, default=-6.0, help="default: -6")
    p.add_argument("--aa-gap-extend", type=float, default=-2.0, help="default: -2")
    p.add_argument("--aa-weight", type=float, default=1.0, help="default: 1")
    p.add_argument("--frameshift-penalty", type=float, default=-100.0, help="default: -100")
    p.add_argument("--misaligned-codon-penalty", type=float, default=-20.0, help="default: -20")
    p.add_argument("--max-occ", type=int, default=16, help="Maximum CDS overlaps per genomic position")
    p.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable post-traceback collapse of compensating opposite indels"
    )

    return p.parse_args()


def main():
    args = parse_args()

    ref = read_single_genbank(args.reference)
    query_name, query_seq = read_single_fasta(args.query)

    G_str = str(ref.seq).upper()
    B_str = query_seq.upper()

    G = encode_nt_seq(G_str)
    B = encode_nt_seq(B_str)

    nt_sub = build_nt_sub_matrix(args.nt_match, args.nt_mismatch)
    aa_sub = build_aa_sub_matrix(args.aa_matrix)

    q_fwd_start, q_rev_end = precompute_query_codon_aas(B_str)
    occ_count, occ_cpos, occ_strand, occ_aa = build_annotation_arrays(ref, max_occ=args.max_occ)

    D, trD, trP1, trQ1, best_score, best_i, best_j = aga_fill_matrices(
        G, B,
        nt_sub, aa_sub,
        occ_count, occ_cpos, occ_strand, occ_aa,
        q_fwd_start, q_rev_end,
        args.nt_gap_open, args.nt_gap_extend,
        args.aa_gap_open, args.aa_gap_extend,
        args.aa_weight,
        args.frameshift_penalty,
        args.misaligned_codon_penalty,
        args.mode == "local"
    )

    aligned_ref, aligned_qry, start_i, start_j = traceback_alignment(
        G_str, B_str, trD, trP1, trQ1, best_i, best_j, args.mode == "local"
    )

    if not args.no_cleanup:
        aligned_ref, aligned_qry = cleanup_alignment(aligned_ref, aligned_qry)

    write_alignment(
        args.out,
        ref.id,
        query_name,
        aligned_ref,
        aligned_qry,
        best_score,
        start_i,
        start_j,
        best_i,
        best_j,
    )

    if args.protein_out:
        write_protein_alignments(
            args.protein_out,
            ref,
            query_name,
            aligned_ref,
            aligned_qry,
        )

    print(f"Reference : {ref.id}")
    print(f"Query     : {query_name}")
    print(f"Mode      : {args.mode}")
    print(f"AGA score : {best_score:.3f}")
    print(f"Output    : {args.out}")
    if args.protein_out:
        print(f"Protein   : {args.protein_out}")
    


if __name__ == "__main__":
    main()
