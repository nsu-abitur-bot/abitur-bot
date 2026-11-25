# Public offline tokenizer for #470

`tokenizer-v41.json.gz` is a deterministic gzip of the official tokenizer from
[deepseek-ai/deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f/static/tokenizers/v41).
Upstream revision: `935817c66f5d6e64838f419692545f06787e485f`.
Uncompressed SHA256: `81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99`.
The upstream MIT notice is retained in LICENSE.

The pinned official `deepseek-recipe==0.2.0` renders V4.1 chat and function
tool framing and encodes the complete prompt. No network lookup or text upload
is performed. Only the reviewed text/non-thinking model contract is accepted;
unsupported parameters, multimodal data, missing encoder or changed artifacts
refuse the request before transport. Provider usage is checked against the
reservation; divergence halts the shared budget. Model updates require a new
reviewed contract, not a silent alias or tokenizer substitution.
