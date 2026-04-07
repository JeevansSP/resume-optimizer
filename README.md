# ATS Beater

Open-source AI-powered resume tailoring service. Upload your PDF resume, paste a job description, and get an ATS-optimized tailored resume compiled to PDF.

Built with FastAPI, Vue 3, Google Gemini, and LaTeX.

**Live at [atsbeater.cydratech.com](https://atsbeater.cydratech.com)**

## How It Works

```mermaid
flowchart LR
    A[Upload PDF] --> B[pdfplumber\nextract text]
    B --> C[Gemini Flash\nstructure profile]
    C --> D[(ResumeInfo\nstored in DB)]
    E[Job Description] --> F[Gemini Pro\ntailor resume]
    D --> F
    F --> G[CustomResumeInfo\nreview & edit]
    G --> H[LaTeX builder\n+ pdflatex]
    H --> I[PDF]
```

**Phase 1 — AI Generation:** Your structured profile + the job description go to Gemini Pro, which produces a tailored, keyword-optimized resume. You can review and edit before proceeding.

**Phase 2 — PDF Compilation:** The tailored resume is converted to LaTeX using a custom document class (`resume.cls`), then compiled to PDF via `pdflatex`.

### Job Status Flow

```mermaid
stateDiagram-v2
    [*] --> PENDING
    PENDING --> GENERATING_RESUME
    GENERATING_RESUME --> RESUME_GENERATED
    RESUME_GENERATED --> GENERATING_PDF
    GENERATING_PDF --> READY
    READY --> [*]
    GENERATING_RESUME --> FAILED
    GENERATING_PDF --> FAILED
```

## Features

- **Resume Roast** — Free AI-powered resume analysis with ATS readiness checklist
- **AI Resume Tailoring** — Gemini Pro tailors your resume for each specific job description
- **LaTeX PDF Generation** — Professional typesetting that passes ATS parsing reliably
- **AI Chat Editor** — Refine your resume through conversation (powered by Google ADK)
- **Credit System** — Daily free credits + purchasable credit packs via Razorpay
- **Multi-tenancy** — Organization labeling with auto-assignment via email domain rules
- **Admin Panel** — Full CRUD for users, tenants, credits, promo codes, transactions
- **Shareable Roast Links** — Share your resume roast results with a public link

## Tech Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI, SQLAlchemy (async), PostgreSQL, Alembic |
| AI | Google `google-genai` SDK, Google ADK (chat agents) |
| PDF | pdflatex + custom `resume.cls`, pdfplumber for extraction |
| Frontend | Vue 3 + Tailwind CSS + Pinia — all via CDN, no build step |
| Auth | Google OAuth 2.0 → JWT |
| Payments | Razorpay (credit packs, time passes) |
| Storage | Google Cloud Storage |
| Package mgr | [UV](https://docs.astral.sh/uv/) |

## Quick Start

### Prerequisites

- **Python 3.12+**
- **Docker** (for PostgreSQL)
- **TeX Live** with `pdflatex`
  - macOS: `brew install --cask mactex`
  - Ubuntu: `apt install texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended lmodern`
- **UV** package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Setup

```bash
# Clone
git clone https://github.com/cydratech/ats-beater.git && cd ats-beater

# Copy environment file and fill in your keys
cp .env.example .env

# Start PostgreSQL
docker compose up -d

# Install dependencies
uv sync --extra dev

# Run database migrations
uv run alembic upgrade head

# Start the server
uv run python -m app.main
```

Open **http://localhost:8000**. Set `DEV_AUTH_BYPASS=true` in `.env` to skip Google OAuth during development.

### Required API Keys

| Key | Where to get it |
|-----|----------------|
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | [Google Cloud Console](https://console.cloud.google.com/apis/credentials) — create OAuth 2.0 credentials |
| `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` | [Razorpay Dashboard](https://dashboard.razorpay.com/) (optional — for payments) |

## Running Tests

```bash
# Unit tests (in-memory SQLite, no external dependencies)
uv run pytest tests/ -v --ignore=tests/integration

# Integration smoke tests (needs running DB + Gemini API key + pdflatex)
INTEGRATION=1 uv run pytest tests/integration/ -v
```

173 unit tests covering models, schemas, API routes, LaTeX builder/sanitizer, JWT handler, credit service, and more.

## Project Structure

```
app/
  main.py                  # FastAPI factory, CORS, exception handlers
  config.py                # Pydantic BaseSettings from .env
  dependencies.py          # Auth (JWT/dev bypass), DB session
  models/                  # SQLAlchemy ORM (User, Profile, Job, Credit, Roast, Tenant)
  schemas/                 # Pydantic schemas (ResumeInfo, CustomResumeInfo, etc.)
  services/
    ai/                    # Gemini inference + prompts + retry
    ocr/                   # PDF text extraction (pdfplumber + Gemini vision fallback)
    latex/                 # LaTeX builder, compiler, sanitizer
    chat/                  # AI chat agents (Google ADK) for resume editing
    profile/               # Profile CRUD + background processing
    job/                   # Job generation (Phase 1 + Phase 2)
    credit/                # Credit balance, deduction, refund, promo codes
    payment/               # Razorpay integration
    storage/               # GCS upload/download
  api/                     # FastAPI route handlers

frontend/
  index.html               # SPA shell (CDN imports, CSS)
  landing.html             # Public landing page
  static/js/app.js         # Entire Vue 3 app (stores, pages, router)

tests/                     # 173 unit tests + integration smoke tests
alembic/                   # Database migrations
resume.cls                 # LaTeX document class
infra/                     # Docker, Cloud Run deploy script, entrypoint
```

## API Endpoints

### Auth
| Method | Path | Description |
|--------|------|-------------|
| GET | `/auth/google/login` | Returns Google OAuth URL |
| GET | `/auth/google/callback` | Exchanges auth code, redirects with JWT |
| GET | `/auth/me` | Current user info |

### Profiles
| Method | Path | Description |
|--------|------|-------------|
| POST | `/profiles/upload` | Upload PDF resume (202, background processing) |
| GET | `/profiles/` | List profiles (paginated) |
| GET | `/profiles/{id}` | Get profile with resume_info |
| PUT | `/profiles/{id}` | Update resume_info |
| DELETE | `/profiles/{id}` | Soft delete |

### Jobs
| Method | Path | Description |
|--------|------|-------------|
| POST | `/jobs/` | Create job (profile_id + job description) |
| POST | `/jobs/{id}/generate-resume` | Trigger AI tailoring (202, deducts credit) |
| POST | `/jobs/{id}/generate-pdf` | Trigger LaTeX compilation (202) |
| GET | `/jobs/{id}/pdf` | Download generated PDF |
| GET | `/jobs/{id}` | Get job details |
| POST | `/jobs/{id}/chat` | Chat with AI to edit resume (SSE stream) |

### Roasts
| Method | Path | Description |
|--------|------|-------------|
| POST | `/roasts/upload` | Upload PDF for AI roast (free, 202) |
| GET | `/roasts/` | List roasts (paginated) |
| GET | `/roasts/shared/{share_id}` | Public shared roast (no auth) |

### Credits & Payments
| Method | Path | Description |
|--------|------|-------------|
| GET | `/credits/packs` | List credit packs (public) |
| GET | `/credits/me` | Balance + daily free + active pass |
| POST | `/credits/redeem-promo` | Redeem promo code |
| POST | `/payments/create-order` | Create Razorpay order |
| POST | `/payments/verify` | Verify payment + credit account |

### Admin
Full CRUD for tenants, users, domain rules, credit packs, time passes, promo codes, and transactions under `/admin/*`. Requires `is_super_admin` flag.

## Admin & Roles

### Setting up a super admin

The first user to sign up won't have admin access. Set it directly in the database:

```sql
UPDATE users SET is_super_admin = true WHERE email = 'your-email@example.com';
```

Once set, the **Admin** tab appears in the sidebar. Super admins can:
- View dashboard KPIs (users, jobs, revenue, LLM usage)
- Manage users (search, assign tenants, grant credits)
- Create/edit credit packs and time passes
- Create/manage promo codes
- View all transactions
- Manage tenants and domain rules

### Multi-Tenancy & Domain Rules

Tenants are organizations (companies, universities) used for labeling — **not data isolation**. All data remains scoped by user.

**How it works:**
1. Create a tenant in the Admin panel (e.g. "MIT", "Google")
2. Add a domain rule mapping an email domain to that tenant (e.g. `mit.edu` → "MIT")
3. When a user signs up via Google OAuth with `@mit.edu`, they're auto-assigned to the "MIT" tenant

**Manual assignment:** Admins can also manually assign any user to a tenant from the Users tab.

**What tenants give you:**
- Organizational labeling in the admin panel
- Tenant name shown on user profiles
- Ability to filter/search users by organization
- Domain-based auto-assignment on signup

```sql
-- Example: Create a tenant and domain rule
INSERT INTO tenants (id, name) VALUES (gen_random_uuid(), 'MIT');
INSERT INTO tenant_domain_rules (tenant_id, domain)
  VALUES ('<tenant-id-from-above>', 'mit.edu');
```

Or do it via the Admin UI → Settings tab → Tenants & Domain Rules.

## Credit System

| Priority | Source | Details |
|----------|--------|---------|
| 1 | Active time pass | Unlimited (no deduction) |
| 2 | Daily free | 3/day (configurable), resets at midnight UTC |
| 3 | Purchased credits | From balance |
| 4 | No credits | 429 error, frontend shows paywall |

Credits are deducted synchronously before generation starts. If generation fails, a refund is issued automatically.

### Promo Codes

Admins can create promo codes from the Admin panel:
- **CREDITS type** — adds N credits to the user's balance
- **TIME_PASS type** — activates a time pass tier (unlimited generations for N days)
- One redemption per user per code
- Optional max total redemptions and expiry date
- Time passes stack: if a user buys a second pass while one is active, the new pass starts at the old expiry

## AI Chat Agents

Both the profile and job pages have AI chat panels powered by [Google ADK](https://google.github.io/adk-docs/) (Agent Development Kit).

### Profile Chat (`profile_editor` agent)
- Reads and edits the user's master profile data (`ResumeInfo`)
- Tools: `get_profile`, `edit_profile` (JSON Patch operations)
- Knows the full product flow — correctly directs users to the Jobs section for PDF downloads
- Will not fabricate UI elements that don't exist

### Job Chat (`resume_editor` agent)
- Reads and edits the tailored resume (`CustomResumeInfo`)
- Tools: `get_resume`, `edit_resume` (JSON Patch operations)
- After edits, auto-recompiles the PDF in the background
- Knows about the fixed LaTeX template — will explain section ordering when asked
- Refuses to generate fake metrics or fabricate experience

### Chat persistence
Chat history is stored via ADK's `DatabaseSessionService` in PostgreSQL (`sessions` and `events` tables). Sessions are keyed by `profile_chat_{id}` or `job_chat_{id}`.

## Resume Roast

Free feature — no credits required. Users upload a PDF and get:
1. **Comedic roast** — AI-generated roast points about the resume
2. **ATS readiness checklist** — 8 criteria (machine readability, contact info, skills, dates, etc.)
3. **Shareable link** — public URL with OG meta tags for social sharing

Roasts use content-based deduplication (SHA-256 hash). Re-uploading the same PDF returns the cached result.

### Share link analytics
Every view of a shared roast link is tracked (`roast_views` table) with:
- IP address, user agent, referer
- Parsed platform (WhatsApp, etc.), OS, browser

## Deployment

### Docker

```bash
docker build -t ats-beater .
docker run -p 8080:8080 --env-file .env ats-beater
```

### Cloud Run

```bash
export GCP_PROJECT_ID=your-project
bash infra/deploy-cloudrun.sh
```

The deploy script handles Artifact Registry, Docker build, push, and Cloud Run deployment. See `infra/deploy-cloudrun.sh` for details.

### Environment

All configuration is via environment variables. See `.env.example` for the full list.

For production, you'll need:
- PostgreSQL instance (Cloud SQL or self-hosted)
- GCS bucket for PDF storage
- Google OAuth credentials with correct redirect URI
- Razorpay keys (optional — for payments)

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | — | PostgreSQL async connection string (`postgresql+asyncpg://...`) |
| `GEMINI_API_KEY` | Yes | — | Google AI API key from [AI Studio](https://aistudio.google.com/apikey) |
| `GEMINI_FLASH_MODEL` | No | `gemini-3-flash-preview` | Model for profile structuring, roasts, and chat |
| `GEMINI_PRO_MODEL` | No | `gemini-3.1-pro-preview` | Model for resume tailoring (higher quality) |
| `GOOGLE_CLIENT_ID` | Prod | — | Google OAuth 2.0 client ID |
| `GOOGLE_CLIENT_SECRET` | Prod | — | Google OAuth 2.0 client secret |
| `JWT_SECRET` | Yes | `change-this-secret` | Secret for signing JWTs (min 32 chars in production) |
| `JWT_ALGORITHM` | No | `HS256` | JWT signing algorithm |
| `JWT_EXPIRY_HOURS` | No | `24` | JWT token expiry in hours |
| `RAZORPAY_KEY_ID` | No | — | Razorpay key ID (skip to disable payments) |
| `RAZORPAY_KEY_SECRET` | No | — | Razorpay key secret |
| `RAZORPAY_WEBHOOK_SECRET` | No | — | Razorpay webhook signature secret |
| `DAILY_FREE_CREDITS` | No | `3` | Free resume generations per user per day |
| `GCS_BUCKET` | No | — | GCS bucket name for PDF storage (skip for local-only) |
| `GCS_CREDENTIALS_PATH` | No | — | Path to GCS service account JSON (uses ADC if omitted) |
| `LATEX_BIN_PATH` | No | `/Library/TeX/texbin` | Directory containing `pdflatex` binary |
| `ENVIRONMENT` | No | `DEV` | `DEV` or `PROD` — affects error verbosity |
| `FRONTEND_URL` | No | `http://localhost:8000` | Used for CORS origins and OAuth redirect |
| `DEV_AUTH_BYPASS` | No | `false` | Set `true` to skip OAuth in development (auto-creates a test user) |
| `RUN_MIGRATIONS` | No | `false` | Set `true` to run Alembic migrations on container startup |

## Pre-flight Check

Verify all external services before deploying:

```bash
uv run python infra/preflight.py
```

Checks PostgreSQL connectivity + schema, Gemini Flash & Pro models, LaTeX compiler, and GCS bucket. Exits with code 0 if all pass, 1 if any fail.

## Troubleshooting

### "LaTeX compilation timed out"
- pdflatex has a 90-second timeout per pass. Image-heavy PDFs or CPU-constrained containers can hit this.
- Fix: increase CPU allocation (2 vCPU recommended) or increase `PDFLATEX_TIMEOUT` in `app/services/latex/compiler.py`.

### "No /Root object! - Is this really a PDF?"
- The uploaded file isn't a valid PDF (might be a .docx or image renamed to .pdf).
- The frontend enforces a 5MB limit and PDF MIME type check.

### "Memory limit exceeded"
- Large/image-heavy PDFs can cause pdfplumber to consume excessive memory during text extraction.
- Fix: increase container memory (2Gi recommended) or reduce the upload size limit.

### LaTeX special characters breaking compilation
- The sanitizer (`app/services/latex/sanitizer.py`) escapes `& % $ # _ { } ~ ^` and common Unicode.
- URLs are handled specially — only `%` is escaped (to `\%`) to prevent LaTeX comment breakage.
- If you encounter a new character that breaks compilation, add it to `_UNICODE_MAP` or `handle_special_chars`.

### Chat returns "A chat request is already in progress"
- Concurrency is managed via in-memory task tracking (`_active_tasks` dict).
- This can happen if a previous request crashed without cleanup. Refreshing the page clears it.

### OAuth redirect mismatch
- Ensure `FRONTEND_URL` matches your domain exactly (including `https://`).
- Add `{FRONTEND_URL}/auth/google/callback` to your Google OAuth authorized redirect URIs.

## Key Design Decisions

- **Two-phase generation** — AI tailoring and PDF compilation are separate. Users can review/edit the AI output before committing to PDF.
- **Background processing** — Profile OCR and job generation run as tracked async tasks with independent DB sessions.
- **Dual extraction** — pdfplumber first (instant), Gemini vision OCR fallback for scanned PDFs.
- **LaTeX over HTML-to-PDF** — Professional typesetting that passes ATS parsing. Custom `resume.cls` handles formatting.
- **No build step frontend** — Vue 3 CDN global build. Just static files served by FastAPI. No Node.js needed.
- **AI chat agents** — Google ADK powers the resume editing chat with tool-calling (read/edit via JSON Patch).

## Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

## License

MIT License. See [LICENSE](LICENSE).

---

Built by [Cydratech](https://cydratech.com)
