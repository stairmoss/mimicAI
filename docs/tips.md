# Tips & Notes

- When cloning **Chinese dialects**, provide both `ref_audio` and `instruct` (e.g., `ref_audio="sichuan.wav", instruct="四川话"`) for more stable dialect output.
- **Min Nan Chinese (闽南语, also known as Hokkien)** can only be synthesized using [Tai-lo romanization](https://en.wikipedia.org/wiki/T%C3%A2i-l%C3%B4) as input; Chinese characters are not supported for Min Nan Chinese in the current model version.
- The model may not reliably generate short audio clips (e.g., 1–2 seconds) without reference audio. If you need to generate short clips, provide reference audio to the model.
