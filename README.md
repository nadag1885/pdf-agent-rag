# 📚 Document Q&A — Private PDF Knowledge Base (RAG)

A local Retrieval-Augmented-Generation app. An **administrator** places approved
PDFs in a private `documents/` folder and indexes them; **users** only get a chat
box to ask questions. Answers are grounded **only** in the indexed PDFs and always
cite the source **filename and page number**. If the documents don't contain the
answer, the app replies:

> This information was not found in the available documents.

**Users cannot upload files.** There is no upload component anywhere in the app.

### 🌐 Bilingual: Arabic and English
The documents and questions can be in **Arabic or English**. A multilingual
embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) is used, so you can ask
in Arabic about English content and vice-versa, and answers are given in the
question's language. Note: retrieval quality for a given passage depends on that
passage extracting as clean text — some scanned or complex-font Arabic pages may
not extract well (see Troubleshooting).

---

## Stack

| Purpose | Technology |
|---|---|
| PDF text extraction | PyMuPDF (`fitz`) |
| RAG orchestration | LangChain |
| Vector database (local, persistent) | ChromaDB |
| Embeddings (free, local) | Hugging Face `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Answer generation | Groq API (`llama-3.3-70b-versatile`) |
| UI | Streamlit |
| Config / secrets | python-dotenv (`.env`) |

---

## ⚠️ Important: this project runs under WSL2 (Ubuntu) on this machine

This Windows machine has **Smart App Control enabled**, which blocks the unsigned
native libraries these packages need (`numpy`, `torch`, PyMuPDF, ChromaDB…). They
run fine inside **WSL2 Ubuntu**, so the app runs there. You still use Windows
normally — you drop PDFs in with Explorer and open the app in your Windows browser.

> If you ever move this project to a machine **without** Smart App Control, you can
> run it directly on Windows with the same commands (just use a normal
> `python -m venv .venv`).

---

## One-time setup

WSL2 + Ubuntu and the Python environment are already set up on this machine:
- Ubuntu distro installed (`wsl --install -d Ubuntu`)
- Virtual environment at `~/pdf_venv` inside Ubuntu
- All dependencies installed

If you ever need to recreate the environment:

```bash
# Inside Ubuntu (open "Ubuntu" from the Start menu):
python3 -m venv --without-pip ~/pdf_venv
curl -sS https://bootstrap.pypa.io/get-pip.py | ~/pdf_venv/bin/python -
cd /mnt/c/Users/moham/pdf_agent_v2
~/pdf_venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
~/pdf_venv/bin/python -m pip install -r requirements.txt
```

### Configure the Groq API key
1. Get a key at <https://console.groq.com> → **API Keys**.
2. Open `.env` in the project root and set:
   ```
   GROQ_API_KEY=your_real_key
   ```
   The `.env` file is git-ignored and is never printed or committed.

---

## Everyday use

### 1. Add PDF files (administrator, in Windows)
Copy approved PDFs into the project's `documents\` folder using **File Explorer**:

```
C:\Users\moham\pdf_agent_v2\documents\
```

Only the administrator does this. Users never add files.

### 2. Build / update the index (administrator)
Open **Windows Terminal / PowerShell** and run:

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/moham/pdf_agent_v2 && source ~/pdf_venv/bin/activate && python scripts/index_documents.py"
```

- First run indexes everything.
- Later runs are **incremental**: only new/changed PDFs are processed and
  removed PDFs are purged — no duplicate chunks.
- Full rebuild: add `--rebuild`. Check what's indexed: add `--status`.

### 3. Start the app (for users)

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/moham/pdf_agent_v2 && source ~/pdf_venv/bin/activate && streamlit run app.py --server.fileWatcherType none"
```

> `--server.fileWatcherType none` is required: Streamlit's hot-reload watcher
> otherwise tries to introspect `torch`/`transformers` modules and floods the
> log with `torchvision` errors. We don't need hot-reload for normal use.

Then open `http://localhost:8501` in your Windows browser.
Users just type questions; each answer shows the source filename and page.

> Tip: you can also open the **Ubuntu** terminal directly and run the commands
> after `bash -lc "…"` without the `wsl -d Ubuntu -- bash -lc` wrapper.

---

## How the "only answer from documents" guarantee works
1. **Relevance guard** — the question is embedded and matched against the indexed
   chunks. If nothing is similar enough, the app returns the "not found" sentence
   **without** calling the language model.
2. **Prompt guard** — the model is instructed to use only the retrieved context
   and to reply with the exact "not found" sentence when the context is
   insufficient. It is told never to use outside knowledge.

---

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `GROQ_API_KEY is missing` | Set `GROQ_API_KEY` in `.env`. |
| "No documents are indexed yet" | Add PDFs to `documents\`, then run the indexer. |
| "No PDF files found" when indexing | The `documents\` folder is empty. |
| `SKIPPED (corrupted / unreadable)` | That PDF is damaged; replace it. Other files still index. |
| `SKIPPED (no extractable text)` | The PDF is a scanned image with no text layer; it needs OCR before it can be indexed. |
| "document index is empty or missing" in the app | Run `python scripts/index_documents.py`. |
| "rate-limited (Groq)" | Wait a few seconds and retry. |
| "Groq API key was rejected" | The key in `.env` is wrong/expired. |
| First query is slow | The embedding model downloads once on first use, then is cached. |
| Arabic answers are weak or "not found" for some passages | Some PDFs store Arabic with fonts that don't map to clean Unicode, so the text extracts garbled and can't be retrieved. Headers/English usually extract fine. Fix: re-create that PDF with a proper text layer, or OCR it with Arabic language data before indexing. |

---

## Security
- `.env`, `api_key.env`, the `vectorstore/` database, and the contents of
  `documents/` are git-ignored.
- The Groq API key is loaded from `.env` and is never printed, logged, or committed.
- If a key was ever committed or shared, rotate it at <https://console.groq.com>.

---

## Project layout
```
pdf_agent_v2/
├── app.py                     # Streamlit chat UI (no upload)
├── scripts/
│   └── index_documents.py     # Admin: build/update/rebuild the index
├── src/rag/
│   ├── config.py              # Paths, models, constants, key loading
│   ├── pdf_loader.py          # PyMuPDF page-by-page extraction + hashing
│   ├── store.py               # Embeddings + Chroma factory
│   ├── indexer.py             # Chunking + change detection + persistence
│   └── qa.py                  # Retrieval + Groq + "not found" guard
├── documents/                 # Admin-only PDFs (git-ignored)
├── vectorstore/               # ChromaDB persistent store (git-ignored)
├── requirements.txt
├── .env                       # GROQ_API_KEY (git-ignored)
└── .env.example
```
