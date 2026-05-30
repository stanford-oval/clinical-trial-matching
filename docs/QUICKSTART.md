# Quickstart Guide

Get SatIR running from scratch in 10 minutes.

## 1. Prerequisites

- Python 3.12+
- Azure OpenAI API access (gpt-4.1 or gpt-5 deployment)
- Java 17 (for Snowstorm SNOMED-CT server)

## 2. Clone and Install

```bash
git clone git@github.com:zikai-zhou/SatIR.git
cd SatIR

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install
pip install -e .
pip install -r requirements.txt
```

## 3. Configure

```bash
# Copy and edit configuration
cp .env.example .env
# Edit .env with your Azure OpenAI endpoint and API key

# Or edit satir.toml for persistent config
```

**Required settings:**
- `OPENAI_ENDPOINT` — your Azure OpenAI deployment URL
- `OPENAI_API_KEY` — your API key

## 4. Start External Services

### Elasticsearch (for entity canonicalization)

```bash
# Download (first time only)
mkdir -p ~/elastic && cd ~/elastic
curl -L -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.15-darwin-aarch64.tar.gz
tar -xzf elasticsearch-7.17.15-darwin-aarch64.tar.gz

# Start
~/elastic/elasticsearch-7.17.15/bin/elasticsearch
```

### Snowstorm SNOMED-CT Server

```bash
# Requires Java 17
export JAVA_HOME=$(/usr/libexec/java_home -v17)

# Start (from SatIR root, snowstorm jar must be available)
java -Xms4g -Xmx4g -jar snowstorm-10.7.0.jar --snowstorm.rest-api.readonly=true
```

## 5. Validate Setup

```bash
source .env
satir setup
```

This checks: API connectivity, service availability, data paths, and dependencies.

## 6. Run the Pipeline

### Compile a trial

```bash
satir compile-trial NCT03362970 --side both
```

### Compile a patient

```bash
satir compile-patient sigir-20141
```

### Index into constraint database

```bash
satir index trial
satir index patient
```

### Retrieve matching trials

```bash
satir retrieve --patient sigir-20141
```

### SMT-based eligibility check

```bash
satir match --trial NCT03362970 --patient sigir-20141
```

## 7. Benchmark

```bash
satir benchmark --warmup 2 --runs 3
```

Expected: ~146ms per patient across 5,446 trials.

## 8. Verify

```bash
# Run the invariant test (requires ground truth fixtures in tests/fixtures/)
python tests/test_retrieval_invariant.py

# Run full end-to-end test
python tests/test_end_to_end.py

# Run exhaustive equivalence test (all 59 patients x 3 modes)
python tests/test_exhaustive_equivalence.py
```

## Configuration Reference

All settings can be configured via `satir.toml` or environment variables (env overrides toml).

Run `satir info` to see current configuration.

| Setting | Env Var | satir.toml | Default |
|---------|---------|------------|---------|
| API endpoint | `OPENAI_ENDPOINT` | `api.endpoint` | (required) |
| API key | `OPENAI_API_KEY` | `api.api_key` | (required) |
| Model | `OPENAI_MODEL` | `api.model` | gpt-4.1 |
| Build directory | `SATIR_BUILD` | `paths.build_dir` | ./build |
| Data directory | `TRIAL_DATA` | `paths.data_dir` | ./dataset/clinical_trial |
| Snowstorm URL | `SNOWSTORM_BASE` | `services.snowstorm_url` | http://localhost:8080 |
| Retrieval scope | -- | `retrieval.scope` | any |
| Important mode | -- | `retrieval.important_mode` | all |
| Alt mode | -- | `retrieval.alt_mode` | act |
| Parallel workers | -- | `retrieval.parallel` | 8 |
| UMLS key | `UMLS_API_KEY` | `keys.umls_api_key` | (optional) |

## Project Website

[satir.genie.stanford.edu](https://satir.genie.stanford.edu/)
