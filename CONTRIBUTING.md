# Contributing to ASR Data Augmentation Pipeline

Thank you for your interest in contributing to the ASR Data Augmentation Pipeline! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/asr-data-augmentation-pipeline.git
   cd asr-data-augmentation-pipeline
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards (see below)

3. Test your changes:
   ```bash
   python tests/test_augmentation.py
   ```

4. Commit your changes with a descriptive message:
   ```bash
   git commit -m "Add feature: brief description"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request on GitHub

## Coding Standards

### Python Style
- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Use type hints where appropriate

### Code Organization
- Keep functions focused and single-purpose
- Maintain the existing directory structure:
  - Core modules in `src/asr_pipeline/`
  - Tests in `tests/`
  - Documentation in `docs/`
  - Scripts in `scripts/`
  - Configuration templates in `config/` and `examples/`

### Documentation
- Update README.md if you add new features
- Add docstrings to new functions/classes
- Update relevant documentation in `docs/` if needed
- Include code examples for new functionality

### Testing
- Add tests for new features
- Ensure existing tests pass
- Test with different configuration options

## Types of Contributions

### Bug Reports
When reporting bugs, please include:
- Python version and OS
- Steps to reproduce the issue
- Expected vs actual behavior
- Relevant configuration and logs

### Feature Requests
When suggesting features, please:
- Explain the use case
- Describe the proposed solution
- Consider backward compatibility

### Code Contributions
We welcome:
- New augmentation techniques
- Performance improvements
- Bug fixes
- Documentation improvements
- Test coverage improvements

### Areas for Contribution
- Additional audio augmentation techniques
- Support for more audio formats
- Improved error handling
- Performance optimizations
- Better progress tracking and logging
- Integration with other ML frameworks
- Additional export formats

## Pull Request Guidelines

1. **Keep PRs focused**: One feature or fix per PR
2. **Write clear descriptions**: Explain what changes you made and why
3. **Update documentation**: Include relevant documentation updates
4. **Add tests**: Ensure your changes are tested
5. **Follow coding standards**: Match the existing code style
6. **Keep commits clean**: Use meaningful commit messages

### Commit Message Format
```
Type: Brief description (50 chars or less)

More detailed explanation if needed (wrap at 72 chars).
Explain what changed and why, not how.

- Bullet points are okay
- Use present tense: "Add feature" not "Added feature"
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Code Review Process

1. Maintainers will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be credited in the commit

## Questions?

- Open an issue for questions about contributing
- Check existing issues and PRs first
- Be respectful and constructive

## License

By contributing, you agree that your contributions will be licensed under the GPL-3.0 License.

Thank you for contributing!
