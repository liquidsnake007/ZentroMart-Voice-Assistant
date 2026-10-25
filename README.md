## Submission usage

Extract the ZIP and open a terminal in the extracted project root. The
submission runner requires the persisted local Qdrant store at
`runtime/qdrant/`. Do not delete or rename this directory. The runner does not
rebuild the vector store and does not require the development KB or dataset
files at runtime.

### Requirements

- Python 3.11 or newer
- Internet access for Gnani, Groq, and Gemini API calls
- Valid API keys

### Configuration

Copy `.env.example` to `.env` and set:

- `GNANI_API_KEY`
- `GROQ_API_KEY`
- `GEMINI_API_KEY`

The configured model is Groq's provider-qualified `openai/gpt-oss-120b` and
`models/gemini-embedding-2-preview`. TTS explicitly uses `timbre-v2.5`.

### Installation

```bash
python -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and add the three API keys before running the agent.

### Run

Place customer audio files in `audio_input/`. Supported formats are WAV, MP3,
FLAC, OGG, M4A, and AAC.

```bash
python run.py --input-dir ./audio_input --output-dir ./sample_outputs
```

Each audio file produces a directory containing `transcript.txt`,
`response.txt`, `response.wav`, `sources.json`, and `metadata.json`.
`run_summary.json` records successful and failed files.

The output files are:

- `transcript.txt` — speech-to-text transcript
- `response.txt` — assistant response
- `response.wav` — synthesized response audio
- `sources.json` — retrieved document chunks and source IDs
- `metadata.json` — providers, decision, and tool metadata

The command also creates `sample_outputs/run_summary.json`, which lists
successfully processed and failed files.

## Expected submission layout

```text
team_name_voiceops.zip
├── README.md
├── .env.example
├── requirements.txt
├── run.py
├── runtime/
│   └── qdrant/
├── src/
├── web/
├── web_demo.py
├── sample_outputs/
├── report.pdf
└── demo.mp4
```

## Development-only evaluation

When the development-only practice files are available, the supplied practice
datasets can be evaluated without audio:

```bash
python evaluate.py --rebuild-index --output evaluation_results.json
python evaluate.py --asr --output evaluation_asr_results.json
```

The evaluator processes conversations in turn order and preserves verified
order state between turns. It spaces live model requests by 2.1 seconds by
default to stay below a 30-requests-per-minute provider limit. Override this
with `GROQ_MIN_REQUEST_INTERVAL` only when the account's limit allows it.

## Dataset validation

```bash
python src/validate_dataset.py
```

The reference date for the supplied synthetic dataset is 20 October 2026.

## Visual demo

The optional visual demo provides a presentation-friendly browser interface
with microphone recording, live voice-intensity animation, audio upload,
conversation history, source chips, and response playback. It uses the same
persisted Qdrant collection and agent pipeline as `run.py`.

Start it from the project root:

```bash
python web_demo.py
```

Open `http://127.0.0.1:8765` in a browser and allow microphone access. The
browser can record a voice query or upload a supported audio file. The demo
server must be running with the same `.env` and `runtime/qdrant/` setup as the
submission runner.
