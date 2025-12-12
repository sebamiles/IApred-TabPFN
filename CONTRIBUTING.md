# Contributing to IApred

Thank you for your interest in contributing to IApred! This document provides guidelines and information for contributors.

## 🚀 Ways to Contribute

### Code Contributions
- **Bug fixes**: Fix issues in the codebase
- **Feature additions**: Implement new functionality
- **Performance improvements**: Optimize existing code
- **Documentation**: Improve documentation and examples

### Data Contributions
- **New antigens/non-antigens**: Add experimentally validated sequences
- **Additional pathogens**: Expand pathogen coverage
- **External validation**: Provide independent test datasets

### Research Contributions
- **Method improvements**: Propose new algorithms or approaches
- **Comparative studies**: Benchmark against new predictors
- **Applications**: Demonstrate use in new immunological contexts

## 🛠️ Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- CUDA-compatible GPU (recommended for TabPFN)

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/IApred-PFN.git
cd IApred-PFN

# Create virtual environment
python -m venv iapred_env
source iapred_env/bin/activate  # On Windows: iapred_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

### Development Dependencies
```txt
pytest>=6.0.0        # Testing framework
black>=21.0.0        # Code formatting
flake8>=3.9.0        # Linting
mypy>=0.900          # Type checking
sphinx>=4.0.0        # Documentation
pre-commit>=2.15.0   # Pre-commit hooks
```

## 📋 Development Workflow

### 1. Choose an Issue
- Check [GitHub Issues](https://github.com/yourusername/IApred-PFN/issues) for open tasks
- Comment on the issue to indicate you're working on it
- Create a new issue if you have a specific contribution in mind

### 2. Create a Branch
```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number-description
```

### 3. Make Changes
- Follow the coding standards outlined below
- Write tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 4. Commit Changes
```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: add new feature description

- What was changed
- Why it was changed
- Any breaking changes
"
```

### 5. Push and Create Pull Request
```bash
# Push your branch
git push origin feature/your-feature-name

# Create a Pull Request on GitHub
```

## 📝 Coding Standards

### Python Style
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Maximum line length: 100 characters
- Use type hints for function parameters and return values

### Code Quality
```python
# Good: Clear variable names, type hints, docstrings
def predict_antigenicity(sequence: str, model_type: str = "tabpfn") -> float:
    """
    Predict antigenicity score for a protein sequence.

    Args:
        sequence: Protein sequence in single-letter code
        model_type: Model to use ('svm' or 'tabpfn')

    Returns:
        Antigenicity score between 0 and 1
    """
    # Implementation here
    pass

# Avoid: Unclear names, no type hints
def pred(seq, mod="tabpfn"):
    # Implementation here
    pass
```

### Documentation
- Use Google-style docstrings
- Document all public functions and classes
- Include parameter descriptions and return values
- Provide usage examples where helpful

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_predictor.py

# Run with coverage
pytest --cov=iapred --cov-report=html

# Run integration tests only
pytest tests/integration/
```

### Writing Tests
- Use `pytest` framework
- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names: `test_predict_single_sequence`

```python
import pytest
from iapred.predictor import predict_antigenicity

def test_predict_antigenicity_basic():
    """Test basic antigenicity prediction functionality."""
    sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWYIKKETG"

    score = predict_antigenicity(sequence)

    assert isinstance(score, float)
    assert 0 <= score <= 1
```

### Test Coverage
- Aim for >80% code coverage
- Test both success and failure cases
- Include edge cases and error conditions
- Test with different input types and formats

## 📊 Performance Benchmarks

When making performance-related changes, ensure:

### Prediction Speed
- TabPFN predictions should be <50ms per sequence
- SVM predictions should be <20ms per sequence
- Batch processing should scale linearly

### Memory Usage
- Peak memory < 4GB for standard datasets
- No memory leaks in repeated predictions
- Efficient handling of large FASTA files

### Accuracy Maintenance
- ROC-AUC degradation < 1% for any changes
- Maintain or improve calibration
- Preserve cross-validation performance

## 🔧 Adding New Features

### Model Integration
To add a new prediction model:

1. **Create model class** in appropriate module
2. **Add feature extraction** if needed
3. **Implement training pipeline** with cross-validation
4. **Add to comparison framework**
5. **Update documentation and tests**

### Dataset Expansion
To add new training data:

1. **Validate experimental evidence** rigorously
2. **Check sequence quality** and completeness
3. **Ensure <90% similarity** to existing data
4. **Maintain class balance** and diversity
5. **Update feature extraction** if needed

## 📚 Documentation

### Updating Docs
```bash
# Build documentation
cd docs
make html

# Serve locally
python -m http.server 8000 -d build/html
```

### README Updates
- Keep main README.md up to date
- Update installation instructions
- Include performance benchmarks
- Add usage examples

## 🔍 Code Review Process

### Before Submitting
- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes without discussion
- [ ] Performance benchmarks maintained

### Review Criteria
- **Functionality**: Does the code work as intended?
- **Code quality**: Is the code clean, readable, and well-documented?
- **Testing**: Are there adequate tests with good coverage?
- **Performance**: Does it maintain or improve performance?
- **Documentation**: Is documentation clear and complete?

## 🚨 Issue Reporting

### Bug Reports
Use the bug report template and include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces
- Minimal reproducible example

### Feature Requests
Use the feature request template and include:
- Problem description
- Proposed solution
- Alternative approaches considered
- Potential impact on users

## 📞 Getting Help

- **Discussions**: Use [GitHub Discussions](https://github.com/yourusername/IApred-PFN/discussions) for questions
- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/yourusername/IApred-PFN/issues)
- **Email**: Contact maintainers for sensitive matters

## 📄 License

By contributing to IApred, you agree that your contributions will be licensed under the same MIT License that covers the project.

## 🙏 Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Acknowledged in release notes
- Cited in future publications
- Invited to author position on related papers

---

Thank you for contributing to IApred and helping advance antigenicity prediction research! 🧬✨

