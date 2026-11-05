# Third-party text in the documentation index

The documentation index (`index.json.gz`, `vectors.npy`) holds passages of the following
works, cut into passages and lightly reformatted (reStructuredText and the specification's
schema tables written as plain text; code examples and figures left out). `braino docs
sources` lists them with their versions.

## BIDS specification 1.11.2

Copyright (c) 2018, BIDS Contributors. The Brain Imaging Data Structure specification is
licensed under the Creative Commons Attribution 4.0 International licence (CC-BY-4.0),
<https://creativecommons.org/licenses/by/4.0/>. Source:
<https://github.com/bids-standard/bids-specification/tree/v1.11.2>; rendered at
<https://bids-specification.readthedocs.io/en/v1.11.2/>.

## MNE-Python 1.13.2 documentation

Tutorials, glossary and implementation notes from
<https://github.com/mne-tools/mne-python/tree/v1.13.2>, under the BSD 3-Clause licence:

Copyright 2011-2025 MNE-Python authors

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of
   conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of
   conditions and the following disclaimer in the documentation and/or other materials
   provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific prior written
   permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

## Embedding model

The vectors were computed with Qwen3-Embedding-0.6B (Apache-2.0,
<https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF>), which `braino model setup`
downloads to embed questions; the model is not part of this index.
