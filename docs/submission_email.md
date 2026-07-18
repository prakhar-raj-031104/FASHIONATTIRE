# Submission email (reply on the original thread from Ritesh)

**To:** ritesh.pallod@glance.com
**Subject:** Re: Glance ML Internship Assignment — Prakhar Raj
**Attach:** `Glance_ML_Assignment_Writeup.pdf`

---

Hi Ritesh,

Thank you for the opportunity. My completed assignment is attached.

- **Write-up (PDF):** attached
- **Codebase:** https://github.com/prakhar-raj-031104/FASHIONATTIRE

**Summary:** I built a natural-language fashion retrieval system over a 3,200-image index.
Because CLIP pools each image into a single embedding, it loses which colour belongs to which
garment — so I added per-garment region decomposition (SegFormer + FashionCLIP) alongside
caption and zero-shot attribute signals, fused into a four-signal ranker.

To verify the compositionality claim rather than assert it, I measured top-10 overlap between a
query and its colour-swapped twin across 12 pairs: vanilla CLIP returns 75.0% identical results,
this system returns 52.5% — a 30% relative improvement in compositional separation. The write-up
also documents where it still falls short and why.

Happy to walk through the design or the results on a call.

Best regards,
Prakhar Raj
