# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 5.11.x  | :white_check_mark: |
| 5.10.x  | :white_check_mark: |
| < 5.10  | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

### How to Report

Please use GitHub's **Private Vulnerability Reporting** feature. This allows you to open a secure, private issue that only repository maintainers can see.

1. Navigate to the [Security tab](https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics/security).
2. Click the **Report a vulnerability** button.
3. In your report, please include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We strictly prefer this method over email. If absolutely necessary, you may contact us at **support@aiwhisperers.com**.

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 60 days

### Disclosure Policy

- We follow coordinated disclosure
- Security fixes are prioritized
- Credit given to reporters (unless anonymity preferred)

## Security Considerations

### Model Weights

- Trained model checkpoints may contain memorized patterns from training data
- Do not train on sensitive or private data without proper anonymization
- Checkpoints should not be shared without review

### Dependencies

We regularly scan for vulnerabilities:

```bash
pip-audit
bandit -r src/
```

### Safe Usage

- Run in sandboxed environments when processing untrusted data
- Do not pickle/unpickle model weights from untrusted sources
- Validate inputs before processing

### Data Handling

- This software processes biological sequence data
- Ensure compliance with relevant data protection regulations
- No personal health information (PHI) should be processed without proper safeguards

## Known Limitations

- This is research software, not production-hardened
- GPU memory handling may expose system information
- Log files may contain training data samples

## Acknowledgments

We thank all security researchers who responsibly disclose vulnerabilities.
