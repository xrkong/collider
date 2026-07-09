"""
dataset/ds — k-file + d3plot → downsampled HDF5 pipeline.

Implements SPEC/SPEC_sampling_reconstruction.md §1-5 (node selection only;
the W reconstruction matrix is a separate concern, see ../k_file_downsample.py).

Module map
----------
constants.py     PID ranges, region/material lookup tables (SPEC §3, Appendix A)
kfile_parser.py  *NODE / *ELEMENT_* parser, fixed-width fallback        (SPEC §2)
regions.py       centerline distance + six sampling-region masks        (SPEC §4)
samplers.py      stride / random / poisson_disk / fps + per-part split  (SPEC §5)
sampling.py      orchestrates regions+samplers into one node set
materials.py     *PART / *MAT_xxx parser → per-part material props
d3plot_io.py     d3plot state-file discovery, time scan, frame stride
build_dataset.py CLI entry point: k-file → sample → d3plot → one .h5
"""
