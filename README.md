# Gener-AI-te — Study Documents & Analysis Code (UCF, 2023)

This repository stores **study-related documents and analysis code** for a research program investigating how **origin labels** (e.g., *“made by a human”* vs *“made by an AI”*, and in some cases *no label*) influence people’s reception of visual images.

Primary outcomes include:
- **Aesthetic judgments**
- **Emotional responses**
- **“Is this art?”** judgments
- **“Do you like it?”** judgments

> **Ethics / Oversight:** These materials correspond to a study conducted in **2023 at the University of Central Florida (UCF)** with **approval from the UCF Institutional Review Board (IRB)**. IRB documentation is included in this repository.

---

## Purpose of the study

Generative AI created new uncertainty about authorship, authenticity, and value in visual media. This study was designed to test whether **people’s evaluations of images shift based on what they are told** about an image’s origin (human vs AI), above and beyond the image’s properties themselves.

At a high level, the study examines:
- **Labeling effects:** Do evaluations differ when an identical (or comparable) image is presented as *Human-made* vs *AI-made* (vs unlabeled)?
- **Attribution vs. origin:** Are responses primarily driven by **presented attribution** (the label shown) rather than **actual origin** (true authorship), particularly when participants cannot reliably discriminate origin perceptually?
- **Individual differences:** Do effects vary as a function of participant-level characteristics (e.g., AI attitudes, art background), when included in the analytic models?

---

## Study design (overview)

Across all waves, the general structure was:

1. **Participant survey battery**
   - Demographics and individual-difference measures (e.g., attitudes toward AI and related constructs, as applicable per wave).
2. **Repeated image evaluation task**
   - Participants view multiple images and provide judgments on the outcomes listed above.
3. **Label manipulation**
   - Images are accompanied by a presented label (e.g., Human / AI / No Label), with details varying by wave.

### Repeated-measures structure
The dataset is typically organized in **long format**, with one row per:
- `ParticipantID × ArtworkID` response instance

This supports mixed-effects modeling with random effects for participants and stimuli (artworks), where applicable.

---

## Multi-wave structure: Studies 1–3

Data collection occurred across **three waves**, which we treat as:

- **Study 1**
- **Study 2**
- **Study 3**

While each study preserves the same core goal (label effects on reception), the specific implementation and refinements may differ by wave (e.g., details of labeling accuracy, interface, measures, or stimulus handling). See the wave-specific materials and scripts in this repo for the precise operationalization.

---

## Cross-study analyses

In addition to study-specific analysis plans/scripts, this repository includes **cross-study (integrated) analysis code** intended to:
- Harmonize variables across Studies 1–3
- Pool data for robustness and generalization checks
- Estimate effects across studies while accounting for study-level differences (e.g., via StudyID indicators and mixed-effects structures)

---

## Repository structure (suggested)

Your exact folder names may differ, but this is the intended organization:

- `irb/`  
  IRB approval/exempt documentation, participant-facing “Explanation of Research,” and protocol-related materials.
- `materials/`  
  Study instruments, recruitment language, and permissible stimulus documentation.
- `analysis/`  
  - `study1/` — Study 1 scripts / notebooks  
  - `study2/` — Study 2 scripts / notebooks  
  - `study3/` — Study 3 scripts / notebooks  
  - `cross_study/` — Pooled / integrated scripts across studies
- `docs/`  
  Analysis notes, codebooks, variable dictionaries, manuscript materials (as appropriate).

---

## Data handling and confidentiality

This repo is intended to store **documents and analysis code**. If data are included:
- Share only **de-identified** data (no direct identifiers)
- Ensure data handling aligns with the IRB-approved protocol and any applicable institutional requirements

---

## How to use

1. Read the IRB documentation in `irb/` for protocol context and participant-facing descriptions.
2. Review `materials/` for instruments and study artifacts used during administration.
3. Run wave-specific analyses in `analysis/study1/`, `analysis/study2/`, and `analysis/study3/`.
4. Use `analysis/cross_study/` for pooled analyses across Studies 1–3.

---

## Contact / attribution

This repository is part of a UCF research effort (2023). If you reuse or adapt any materials or code, please provide appropriate attribution consistent with academic norms and any licensing or sharing constraints specified in this repository.
