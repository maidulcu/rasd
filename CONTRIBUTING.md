# Contributing to Rasd

Thanks for your interest in contributing to Rasd! This document provides guidelines and steps for contributing.

## How to Contribute

### 1. Fork the Repository

```bash
# Fork via GitHub UI, then clone
git clone https://github.com/YOUR_USERNAME/rasd.git
cd rasd
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Setup database
cp .env.example .env
createdb video_intelligence
alembic upgrade head
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Changes

- Follow existing code style
- Add tests for new features
- Update documentation if needed

### 5. Commit

```bash
git add .
git commit -m "feat: add your feature description"
```

Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation
- `refactor:` — Code refactoring
- `test:` — Adding tests
- `chore:` — Maintenance

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

---

## Reporting Bugs

Use [GitHub Issues](../../issues) with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Screenshots if applicable
- Environment details (OS, Python version, hardware)

---

## Suggesting Features

Open an issue with:
- Feature description
- Use case / problem it solves
- Proposed implementation (if any)

---

## Code Style

- Python 3.12+
- Follow PEP 8
- Use type hints
- Docstrings for public functions

---

## Testing

```bash
pytest
```

Ensure all tests pass before submitting PR.

---

## Questions?

Open a discussion or reach out to [Dynamic Web Lab](https://dynamicweblab.com).
