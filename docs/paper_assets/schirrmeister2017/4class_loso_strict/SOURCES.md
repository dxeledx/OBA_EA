# Schirrmeister2017 (HGD) — 4-class — LOSO (strict)

All assets in this folder are derived from the following run folder (same protocol / same trials):

- `outputs/20260117/4class/loso4_schirr2017_0train_rs50_ea_fbcsp_stack_prefdisagree037_v1/`

Notes:
- Protocol: cross-subject LOSO on MOABB `Schirrmeister2017` with `--sessions 0train`.
- Preprocessing: `moabb` pipeline, bandpass 8–30 Hz, `--resample 50`, epoch `tmin=0.5s`, `tmax=3.5s`.
- Candidate family in our run: `ea,fbcsp` with `--oea-zo-selector prefer_fbcsp` and `--stack-safe-fbcsp-max-pred-disagree 0.37`.
