# Overview

This is a real estate underwriting model application designed to analyze investment property deals. The system calculates key financial metrics like Internal Rate of Return (IRR) and Debt Service Coverage Ratio (DSCR) to evaluate property investment viability. The application provides both a web-based GUI using Streamlit and a command-line interface for different use cases, including automated testing and CI integration.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Application Structure
The system follows a modular Python architecture with dual interface support:

**Dual Interface Design**: The application provides both a Streamlit web GUI and CLI interface, allowing users to interact through their preferred method. The web interface offers an interactive experience for detailed analysis, while the CLI provides quick calculations and automated testing capabilities.

**Financial Calculation Engine**: Core underwriting logic built around standard real estate investment metrics (IRR, DSCR, cash flows). The model uses NumPy for numerical computations and Pandas for data manipulation, ensuring accurate financial calculations.

**Self-Testing Framework**: Built-in validation system that runs sanity checks on calculations using realistic baseline scenarios. The tests verify that IRR is positive (equity preservation) and minimum DSCR stays above 1.0 (debt coverage requirements).

**Graceful Degradation**: The application handles optional dependencies elegantly - if Streamlit is not available, it falls back to CLI-only mode without breaking core functionality.

## Data Processing
**Input Handling**: Supports various input methods including manual parameter entry through the web interface and programmatic input for automated scenarios.

**Calculation Pipeline**: Sequential processing of deal parameters through financial formulas to generate comprehensive investment analysis reports.

**Output Generation**: Produces formatted summaries and detailed breakdowns suitable for both human review and automated processing.

# External Dependencies

## Core Python Libraries
- **NumPy**: Numerical computations for financial calculations and array operations
- **Pandas**: Data manipulation and financial time series analysis

## Optional Web Framework
- **Streamlit**: Web-based user interface for interactive deal analysis (degrades gracefully if not installed)

## Standard Library Dependencies
- **sys**: Command-line argument processing and system interaction
- **datetime**: Date handling for deal timelines and cash flow projections
- **io.BytesIO**: File handling and data export capabilities
- **typing**: Type hints for code clarity and IDE support

The application is designed to run with minimal external dependencies, making it suitable for deployment in various environments from local development to CI/CD pipelines.