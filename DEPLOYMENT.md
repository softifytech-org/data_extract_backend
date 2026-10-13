# Azure Production Readiness

## 2026-10-13 update - App Service/Static Web Apps hosting prep + submit durability

**Correction to this document's own earlier claims:** sections 1-8 below describe an
earlier pass that assumed API-key multi-user auth would ship. That direction was
reversed later in the same project - the application is intentionally **fully open,
no authentication anywhere** (see `app/api/batches.py`'s module docstring;
`app/core/auth.py`'s `get_current_user_optional` is kept only as a stable stub that
always returns `None`). Ignore the auth-related rows in sections 1/2/4 below; the
rest (Blob Storage status, cost shape, queue/DB status) is still accurate.

Since that pass, two more things shipped, both local-only, nothing deployed:

- **Single-document submission XML is now durable the same way batch XML already
  was.** `POST /api/documents/{id}/submit` writes the legacy dump-format XML to
  Blob Storage (when configured) via the same `FileStore.save_xml`, records a
  `document_submissions` row (`storage_key`, `backend`, `record_count`) **only
  after** that write succeeds, and `GET /api/documents/{id}/submit/download`
  reads from that tracked location (falling back to the pre-existing local file
  for submissions made before this change). Verified live against the real
  `softifyocrfiles2026` account, not just unit-tested.
- **Hosting prep for Azure App Service (Linux) + Azure Static Web Apps:**
  `backend/Dockerfile` + `backend/.dockerignore` (untested - no Docker in this
  dev environment, see the blockers list in chat), `frontend/staticwebapp.config.json`
  (SPA fallback routing), and `app/main.py`'s CORS regex now also allows
  `*.azurestaticapps.net` alongside `*.vercel.app`.
- Two new tests (`backend/tests/test_submit_storage_failure.py`) force a Blob
  Storage write failure during both the per-document and batch submit endpoints
  and assert a 502 **and** that no submission row is ever recorded - confirming
  "recorded only after the durable write succeeds" by test, not just by reading
  the code.
- Full suite: **80/80 backend tests pass**; `npm run build` (frontend) succeeds clean.

**Still true, unchanged:** no Postgres, no Azure Storage Queue, no new Azure
resources were created. The job queue and all record/batch state remain local
SQLite/JSON - see the restart-risk note the chat report restates explicitly.

---

Status: documentation and local preparation. Nothing in this document has been deployed. Two real Azure resources now back this app directly: the existing Document Intelligence resource (S0 Standard, OCR) and an Azure Storage account (`softifyocrfiles2026`, two private Blob containers - `uploaded-images` and `generated-xml`) that the user created and connected on 2026-10-13. Both were verified against the real services (not simulated) - see section 1a. No other Azure resources exist; nothing was deployed, and no billing/tier changes were made to either resource.

## 1a. Azure Blob Storage - what's real now

As of 2026-10-13, uploaded batch images and each batch's combined submission XML are genuinely stored in Azure Blob Storage, not local disk, whenever `AZURE_STORAGE_CONNECTION_STRING`/`AZURE_STORAGE_ACCOUNT_NAME`/`AZURE_UPLOAD_CONTAINER`/`AZURE_XML_CONTAINER` are all set (see `backend/app/services/storage/blob_store.py`). This was verified directly against the real account, not assumed:

- A live end-to-end run (real registration, a real 2-image batch, real Azure Document Intelligence OCR, real approval, real batch submit) produced 2 real blobs in `uploaded-images` and 1 real combined XML blob in `generated-xml` - confirmed by listing the real containers directly via the Azure SDK, not just trusting the app's own API responses.
- The backend process was then killed and restarted (a real OS-level restart, not a simulation), and the same batch/images/approvals/submission record were still fully present and correct afterward.
- All test blobs created during this verification were deleted again afterward - nothing created during this check was left behind in the real account.
- A separate unit-level check confirmed the "missing blob" and "transient failure" paths behave correctly against a mocked SDK (fails fast on a genuine 404, retries with backoff on a transient error, gives up cleanly after 3 attempts) - see `backend/tests/test_blob_store.py`.

**Pre-existing local data is unaffected.** Every document uploaded before this change has no `storage_backend` field; `SourceFileInfo.storage_backend` defaults to `"local"` for those records specifically so they keep resolving through local disk forever, regardless of what new uploads use (see `get_file_store_for_record`). Nothing old was migrated or touched.

This applies to BOTH upload paths, not just batch: the single-document legacy `/upload` endpoint (`app/api/documents.py`) also now writes through the same `FileStore`, so a lone image uploaded outside the batch flow also lands in Blob Storage when configured (under a `legacy/` blob-name prefix, since it has no batch to scope under).

**Still not covered by Blob Storage:** the per-document ad-hoc Excel/XML preview exports (`/export/excel`, `/export/xml` - a different, self-describing schema, regenerated on demand, not the durable submission record) and the single-document legacy submit/download flow's local XML file (`data/submissions/<document_id>.xml`, from the pre-batch `/submit` endpoint) remain on local disk, by design - they are not the "generated XML" this integration targeted, and changing them was out of scope for this pass. The NEW batch-level submit endpoint's combined XML (`/api/batches/{id}/submit`) is the one that goes to Blob Storage.

## 1. Candidate architecture

| Concern | Candidate Azure service | Why | Current local stand-in |
|---|---|---|---|
| Frontend hosting | Azure Static Web Apps | Vite produces a static build; SWA has a free tier and built-in CI from a repo, closest match to what Vercel already does | Vercel (already working) |
| Backend API | Azure App Service (Linux, Python) | FastAPI runs as a standard ASGI app; App Service needs no container work for a plain Python app | Render (already working) |
| Background jobs | Azure Storage Queue | Simplest durable queue option for this job volume (≤2,000 images/batch-day at 50 users); Service Bus is justified only if you need topics/sessions/dead-lettering beyond what's built here already | **Local SQLite-backed queue** (`app/services/storage/app_db.py`) - see note below. **Still not Azure.** |
| Uploaded images + generated XML | Azure Blob Storage | Durable, survives app restarts/redeploys, supports SAS URLs for scoped, time-limited access | **Real Azure Blob Storage, now configured and verified** (`app/services/storage/blob_store.py` - see section 1a). Local disk remains the fallback only when unconfigured (dev/test). |
| Records/batches/users database | Azure Database for PostgreSQL (Flexible Server) | Matches the project's own stated long-term plan (see `json_storage.py`'s module docstring, written before this session); SQLite's per-file-lock model doesn't scale to 50 concurrent authenticated users safely | **SQLite** (`app/services/storage/app_db.py`, `data/submissions.db`) + **JSON-file-per-document** (`JsonDocumentRepository`) - **still not Azure.** |
| OCR | Azure AI Document Intelligence (existing resource, S0) | Already in use, already measured, already paid for | Same - no change |

**Status, precisely:** uploaded images and generated submission XML are now genuinely in Azure Blob Storage (section 1a) - that row is done, not local-only. The job queue and the records/batches/users database are still local SQLite/JSON, not Azure, for the same reason as before: no Azure Storage Queue or PostgreSQL credentials were provided, and provisioning paid resources wasn't authorized. What exists for those two is a genuinely durable LOCAL substitute (SQLite-backed queue with retries/backoff/idempotency/crash-recovery, verified with real automated tests - see `backend/tests/`), built behind clean interfaces so swapping in the real Azure services later is a drop-in implementation change, not a redesign.

## 2. What's implemented vs. what's still local-only

| Requirement | Status |
|---|---|
| Durable job queue (retry, backoff, idempotency, crash recovery) | **Implemented and tested locally** (SQLite). Not yet backed by Azure Storage Queue. |
| Multi-user auth and batch ownership isolation | **Implemented and tested locally** (API-key auth, SQLite users/batches). No OAuth/SSO. |
| Async batch processing (no long-running HTTP requests) | **Implemented and tested locally.** |
| Durable file storage for uploads/generated XML | **Implemented against REAL Azure Blob Storage** (`softifyocrfiles2026`, containers `uploaded-images`/`generated-xml`) - verified with a live end-to-end run plus a real process restart (section 1a). Local disk remains only the dev/test fallback when Azure Storage isn't configured. |
| Durable, scalable database | **Not implemented** - SQLite/JSON files, single-instance only, no real concurrent-write safety beyond SQLite's own file locking. |
| Azure Blob SAS URLs for scoped downloads | **Not implemented** - every download is proxied through an authenticated backend endpoint instead (the containers are private; no blob URL, SAS or otherwise, is ever given to the frontend). This is a deliberate, more conservative choice, not an oversight - see the security section of the final report. |

## 3. Environment configuration

See `.env.example` (repo root) for backend variables and `frontend/.env.example` for the frontend. Four new variables were added this pass: `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_ACCOUNT_NAME`, `AZURE_UPLOAD_CONTAINER`, `AZURE_XML_CONTAINER` - all four live in `.env` (gitignored, never committed) with real values for local dev, and `.env.example` documents them with empty placeholders only. The connection string IS a real secret (it carries the storage account key) - treat it exactly like `DOCUMENTINTELLIGENCE_API_KEY`: environment-variable-only in any real deployment, never in a frontend build, never logged. The API-key auth system separately stores only a SHA-256 hash of each user's key, never the key itself.

**Never put these in a frontend environment variable (browser-exposed):** `DOCUMENTINTELLIGENCE_API_KEY`, `AZURE_STORAGE_CONNECTION_STRING`, any future database connection string, any user's API key.

## 4. Deployment checklist (backend, App Service)

1. Provision an Azure Database for PostgreSQL Flexible Server (not done - requires your decision on tier/region/cost).
2. ~~Provision an Azure Storage account with a Blob container (uploads) and a Storage Queue (jobs)~~ - **the Storage account and both Blob containers are done** (`softifyocrfiles2026`); a Storage Queue for the job queue is still not provisioned (the local SQLite queue remains the active implementation).
3. Migrate `AppDatabase`'s job queue and the records/batches/users data to Postgres + Storage Queue (not done - scoped out this session per the "don't create paid resources beyond what's authorized" rule; the code is structured so this is a swap of the storage-layer modules, not the API layer). The image/XML half of this item is already done - see item 2.
4. Set App Service environment variables: `DOCUMENTINTELLIGENCE_ENDPOINT`, `DOCUMENTINTELLIGENCE_API_KEY`, `FRONTEND_ORIGIN`, `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_ACCOUNT_NAME`, `AZURE_UPLOAD_CONTAINER`, `AZURE_XML_CONTAINER`, plus a Postgres connection string once that's provisioned.
5. Set the App Service startup command to `uvicorn app.main:app --host 0.0.0.0 --port 8000` (App Service sets `$PORT`/`WEBSITES_PORT` conventions slightly differently from Render - confirm the exact variable name in the Azure Portal before first deploy).
6. Point health checks at `GET /api/health` (liveness) and `GET /api/ready` (readiness) - both already implemented and tested this session.
7. Enable "Always On" (App Service setting) so the background worker (started in `app/main.py`'s lifespan) isn't suspended on idle the way Render's free tier suspends it.
8. Confirm CORS (`FRONTEND_ORIGIN`) includes the real Static Web Apps domain.

## 5. Deployment checklist (frontend, Static Web Apps)

1. Build command `npm run build`, output directory `dist` (same as the existing Vercel config).
2. Set `VITE_API_BASE_URL` to the App Service URL.
3. No secrets belong in this app at all - it only ever calls the backend.

## 6. Rollback and recovery plan

- **Backend**: App Service keeps prior deployment slots; rolling back is a slot swap, not a rebuild. Document this requires the "Standard" tier or above (not Free/Basic) once you're actually configuring it - not verified here.
- **Database**: once migrated to Postgres, use Azure's built-in point-in-time restore (enabled by default on Flexible Server) - not configured or tested here, since no Postgres instance was provisioned.
- **Current local SQLite files** (`app.db`, `submissions.db`, per-document JSON): back up by copying the `data/` directory. No automated backup exists yet - this is a real gap for any actual production use, not just an Azure-migration detail.

## 7. Cost estimate (rough, assumptions stated)

| Resource | Tier | Est. monthly cost | Assumption |
|---|---|---|---|
| App Service (backend) | B1 Linux | ~$13 | Single instance, no autoscale |
| Static Web Apps (frontend) | Free tier | $0 | Within free tier's bandwidth/build limits |
| PostgreSQL Flexible Server | Burstable B1ms | ~$12-26 | Varies by region (see pricing research done earlier this session) |
| Blob Storage | Standard LRS, Hot tier | ~$2-5 | **Already live** (`softifyocrfiles2026`) - low volume (a few thousand images/exports) keeps this cheap; actual cost depends on real usage, not this estimate |
| Storage Queue | Standard | <$1 | Not yet provisioned - estimate for when the job queue migrates off local SQLite |
| Document Intelligence | S0, pay-per-page | Variable | ~$10/1,000 pages (Layout model), already in use |
| **Rough total** | | **~$40-60/month** + OCR usage | Planning estimate only, not a quote - confirm against the Azure pricing calculator before budgeting |

## 8. Production prerequisites that cannot be verified locally

- Real Azure Storage Queue behavior - only the local SQLite substitute was tested; Blob Storage itself (uploads + generated XML) WAS verified against the real account (section 1a), so this item now only applies to the job queue.
- Blob Storage SAS URLs - not used at all (downloads are proxied through the backend instead), so there's nothing to verify here; if a future design wants direct-to-blob downloads, SAS generation/expiry would need its own verification pass.
- Real PostgreSQL concurrent-write behavior under 50 simultaneous users - only SQLite was tested, and SQLite's concurrency model is materially different.
- Real App Service cold-start/scaling behavior.
- Real network latency/CORS behavior between actual Static Web Apps and App Service domains.
- Genuine 50-concurrent-user load behavior against real infrastructure (see the load-test plan - only a local, smaller-scale run was executed).
- Azure Blob Storage behavior specifically AT 50-concurrent-user scale (the verification in section 1a was a single small batch, not a load test) - the retry/backoff logic is unit-tested for correctness (`test_blob_store.py`) but not load-tested against the real account.
