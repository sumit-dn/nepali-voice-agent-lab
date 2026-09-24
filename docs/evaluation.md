# Evaluation methodology

## Text normalisation (ASR scoring and LLM checks)
The same steps are applied to reference and hypothesis, and recorded in every run:
1. NFC
2. drop ZWJ/ZWNJ
3. Devanagari digits → ASCII
4. drop digit-group commas
5. Unicode punctuation and symbols (including । ॥) → space
6. lowercase
7. collapse whitespace

The implementation avoids regex `\w`, because Python's `\w` does not match Devanagari vowel signs or the virama.

## ASR metrics
- **WER / CER** are corpus-level: sum of edits divided by sum of reference length. They are computed from
  per-sample counts (S/D/I, char edits) stored in the results. CER counts Unicode code points, so a matra is one unit.
- **Critical entities** are annotated per sample (`entities`):
  - Numeric types (phone, OTP, account numbers, numbers, currency) are matched as digit strings with
    digit boundaries, ignoring script, spacing and grouping.
  - Text types (names, addresses, dates, times) are matched as normalised phrases.
  - Accuracy = hits / total.
  - **Limitation:** number *words* do not match digits. Fixing that needs Nepali inverse text normalisation
    (planned), and until then these misses are reported, not hidden.
- **Mixed-language recall**: the share of Latin-script reference words that appear verbatim in the hypothesis.
- **Error tags** are derived from the word alignment: substitution, deletion, insertion, number,
  english_word, code_switching (English reference transcribed in Devanagari), devanagari, and missed:<entity>.
  `reports/asr-error-analysis.md` shows examples for each tag.
- **Latency / RTF** cover model inference only (load time is separate). RAM, CPU, VRAM and GPU utilisation
  are sampled at 4 Hz for each model run.

## LLM metrics
- Deterministic validators run per case; pass rates are reported per check type (tool_name, tool_args,
  contains, script, brevity…).
- TTFT, total latency and tokens per second are reported.
- A **hallucination probe** asks about something the knowledge base does not contain; passing requires an
  "I don't know" or transfer answer.
- Subjective qualities (fluency, naturalness, conciseness) are **flagged for human review**. There is no
  LLM judge.

## TTS metrics
- Latency, audio duration, RTF, sample rate and resources.
- **ASR-judge CER**: an ASR model transcribes the TTS output and the transcript is compared with the input
  text. This is an intelligibility proxy. It is confounded by the judge's own errors, so compare TTS models
  under the *same* judge only.
- **Human rating (protocol; tooling is a remaining task).**
  - Criteria, each on a 1–5 scale: naturalness, pronunciation, clarity, prosody, accent, intelligibility,
    mixed-language quality.
  - Setup: blind and randomised clip order, same texts for every system, at least 8 raters, anonymous
    rater ids, and hidden reference/anchor clips.
  - Report mean ± 95% CI per criterion. Call it "MOS" only if the P.800-style conditions were met.

## Synthetic audio
Scores computed on TTS-generated audio (`dataset synth`) only validate the plumbing. Use real, consented
recordings (and public sets such as FLEURS `ne_np`) for any model decision. Most Nepali fine-tunes were
trained on OpenSLR-54, so don't use it as their test set.
