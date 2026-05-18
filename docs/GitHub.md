# GitHub Guide

## Repository Setup

### 1. Create a new repository on GitHub

Go to https://github.com/new and create a repository named `neuro-sl-optimizer`.

### 2. Push the local project

```bash
cd neuro-sl-optimizer
git init
git add .
#git reset
#git rm --cached old_versions
git status
git remote add origin https://github.com/alep0/neuro-sl-optimizer.git
git remote set-url origin git@github.com:alep0/neuro-sl-optimizer.git
git remote -v
git commit -m "feat: initial project structure and PSO refactor"
git branch -M main
git push -u origin main

ls -al ~/.ssh
ssh-keygen -t ed25519 -C "aaaguado@ifisc.uib-csic.es"

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

cat ~/.ssh/id_ed25519.pub
ssh -T git@github.com

git clone git@github.com:alep0/neuro-sl-optimizer.git
git clone https://github.com/alep0/neuro-sl-optimizer.git

```

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, CI-passing code |
| `develop` | Integration branch for features |
| `feature/<name>` | Individual feature branches |
| `hotfix/<name>` | Urgent bug fixes on `main` |

---

## CI/CD — GitHub Actions

The workflow is defined in `.github/workflows/ci.yml` and runs on every
push and pull request to `main` and `develop`.

### Workflow steps

1. **Lint** — `ruff` checks coding style.
2. **Format** — `black --check` verifies formatting.
3. **Build C++ extension** — `python setup.py build_ext --inplace`.
4. **Unit tests** — `pytest validations/ -v --cov`.
5. **Validate config** — `python validations/validate_config.py`.

### Viewing CI results

Navigate to the **Actions** tab of your repository on GitHub.

### Required secrets

None by default.  If you later add artifact upload or PyPI publishing,
set `PYPI_TOKEN` in *Settings → Secrets and variables → Actions*.

---

## Pull Request Checklist

Before opening a PR:

- [ ] All tests pass locally: `pytest validations/ -v`
- [ ] Code formatted: `black source/ scripts/ validations/`
- [ ] No lint errors: `ruff check source/ scripts/ validations/`
- [ ] `CHANGELOG.md` updated (if applicable)
- [ ] Docstrings updated for new/changed public functions

---

## Releases

1. Bump the version in `setup.py`.
2. Tag the commit: `git tag v2.1.0 && git push origin v2.1.0`
3. Create a GitHub Release from the tag.

---

## Issue Templates

Consider adding `.github/ISSUE_TEMPLATE/bug_report.md` and
`feature_request.md` for structured contributor feedback.

---

## Recommended GitHub Settings

| Setting | Value |
|---|---|
| Default branch | `main` |
| Branch protection on `main` | Require status checks to pass |
| Squash merging | Enabled (keeps history clean) |
| Auto-delete head branches | Enabled |
